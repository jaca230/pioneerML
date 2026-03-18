from __future__ import annotations

import shutil
import tempfile
from collections.abc import Mapping
from typing import Any

import pytorch_lightning as pl
import torch

from pioneerml.common.integration.zenml.utils import detect_available_accelerator
from .earlystopping import build_early_stopping_callback
from ...utils import (
    build_loader_bundle,
    maybe_compile_model,
    merge_nested_dicts,
)


def resolve_effective_training_config(*, config: dict[str, Any], runtime_overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    return merge_nested_dicts(base=dict(config), override=runtime_overrides)

def build_training_module(
    *,
    objective_adapter: Any,
    cfg: dict[str, Any],
    context: str,
):
    model = objective_adapter.build_model(
        model_cfg=dict(cfg.get("model") or {}),
        compile_cfg=None,
        context=context,
    )
    model = maybe_compile_model(model, cfg.get("compile"), context=context)
    return objective_adapter.build_module(model=model, train_cfg=cfg)


def build_train_val_providers(*, loader_factory, cfg: dict[str, Any]):
    train_provider, train_params, _ = build_loader_bundle(
        loader_factory=loader_factory,
        cfg=cfg,
        purpose="train",
        default_shuffle=True,
    )
    val_provider, val_params, _ = build_loader_bundle(
        loader_factory=loader_factory,
        cfg=cfg,
        purpose="val",
        default_shuffle=False,
    )
    return train_provider, val_provider, train_params, val_params


def validate_train_val_providers(*, step: Any, train_provider, val_provider) -> None:
    if not train_provider.include_targets or not val_provider.include_targets:
        raise RuntimeError(
            f"{step.__class__.__name__} expects train/val loaders with targets enabled."
        )


def restore_eager_model_if_compiled(module) -> None:
    model_obj = getattr(module, "model", None)
    if model_obj is not None and hasattr(model_obj, "_orig_mod"):
        module.model = model_obj._orig_mod


def fit_module_with_loaders(
    *,
    module,
    train_loader,
    val_loader,
    max_epochs: int,
    grad_clip: float | None,
    trainer_kwargs: dict | None,
    early_stopping_cfg: dict | None,
):
    merged_trainer_kwargs = dict(trainer_kwargs or {})
    if grad_clip is not None:
        merged_trainer_kwargs.setdefault("gradient_clip_val", float(grad_clip))

    es_cfg = dict(early_stopping_cfg or {})
    es_cfg_inner = dict(es_cfg.get("config") or {})
    monitor = str(es_cfg_inner["monitor"])
    mode = str(es_cfg_inner["mode"])

    callbacks = list(merged_trainer_kwargs.get("callbacks") or [])
    ckpt_dir = tempfile.mkdtemp(prefix="pioneerml_best_ckpt_")
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best",
        monitor=monitor,
        mode=mode,
        save_top_k=1,
        save_last=False,
        auto_insert_metric_name=False,
    )
    callbacks.append(ckpt_callback)
    es_callback = build_early_stopping_callback(early_stopping_cfg=early_stopping_cfg)
    if es_callback is not None:
        callbacks.append(es_callback)
    if callbacks:
        merged_trainer_kwargs["callbacks"] = callbacks

    try:
        detected_accelerator, detected_devices = detect_available_accelerator()
        accelerator = merged_trainer_kwargs.pop("accelerator", detected_accelerator)
        devices = merged_trainer_kwargs.pop("devices", detected_devices)
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            max_epochs=int(max_epochs),
            enable_checkpointing=True,
            logger=False,
            **merged_trainer_kwargs,
        )
        trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

        best_model_path = str(getattr(ckpt_callback, "best_model_path", "") or "")
        if best_model_path:
            best_ckpt = torch.load(best_model_path, map_location="cpu")
            state_dict = best_ckpt.get("state_dict", best_ckpt)
            module.load_state_dict(state_dict, strict=True)
    finally:
        restore_eager_model_if_compiled(module)
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    return module
