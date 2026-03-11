from __future__ import annotations

import logging
import shutil
import tempfile
from collections.abc import Mapping
from typing import Any

import pytorch_lightning as pl
import torch

from pioneerml.common.integration.zenml.utils import detect_available_accelerator
from .earlystopping import build_early_stopping_callback


def apply_compile_runtime_defaults(cfg: dict) -> None:
    matmul_precision = str(cfg["matmul_precision"]).strip().lower()
    if matmul_precision in {"high", "medium", "highest"}:
        torch.set_float32_matmul_precision(matmul_precision)

    capture_scalars = cfg.get("capture_scalar_outputs")
    if capture_scalars is not None and hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "config"):
        torch._dynamo.config.capture_scalar_outputs = bool(capture_scalars)

    skip_dynamic_cudagraphs = cfg.get("cudagraph_skip_dynamic_graphs")
    if (
        skip_dynamic_cudagraphs is not None
        and hasattr(torch, "_inductor")
        and hasattr(torch._inductor, "config")
        and hasattr(torch._inductor.config, "triton")
    ):
        torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = bool(skip_dynamic_cudagraphs)

    max_autotune = cfg.get("max_autotune")
    if hasattr(torch, "_inductor") and hasattr(torch._inductor, "config"):
        if hasattr(torch._inductor.config, "max_autotune"):
            torch._inductor.config.max_autotune = bool(max_autotune)
        max_autotune_gemm = cfg.get("max_autotune_gemm")
        if hasattr(torch._inductor.config, "max_autotune_gemm"):
            torch._inductor.config.max_autotune_gemm = bool(max_autotune_gemm)

    inductor_log_level = cfg.get("inductor_log_level")
    if inductor_log_level is not None and hasattr(torch, "_logging") and hasattr(torch._logging, "set_logs"):
        level = getattr(logging, str(inductor_log_level).upper(), logging.ERROR)
        torch._logging.set_logs(inductor=level)


def maybe_compile_model(model, compile_cfg: dict | None, *, context: str = "train"):
    cfg = dict(compile_cfg or {})
    if not bool(cfg["enabled"]):
        return model
    if not hasattr(torch, "compile"):
        print(f"[{context}] torch.compile unavailable; using eager mode.")
        return model

    apply_compile_runtime_defaults(cfg)
    mode = str(cfg["mode"])
    dynamic = cfg.get("dynamic")
    backend = cfg.get("backend")
    kwargs = {"mode": mode}
    if dynamic is not None:
        kwargs["dynamic"] = bool(dynamic)
    if backend:
        kwargs["backend"] = str(backend)
    try:
        return torch.compile(model, **kwargs)
    except Exception as exc:
        print(f"[{context}] torch.compile failed ({exc}); using eager mode.")
        return model


def merge_nested_dicts(*, base: dict[str, Any], override: Mapping[str, Any] | None) -> dict[str, Any]:
    out = dict(base)
    if override is None:
        return out
    for key, value in dict(override).items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = merge_nested_dicts(base=dict(out[key]), override=value)
        else:
            out[key] = value
    return out


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
    loader_cfg = dict(cfg["loader_config"])
    base_cfg = dict(loader_cfg["base"])
    train_params = {**base_cfg, **dict(loader_cfg["train"])}
    val_params = {**base_cfg, **dict(loader_cfg["val"])}
    train_provider = loader_factory.build_loader(loader_params=train_params)
    val_provider = loader_factory.build_loader(loader_params=val_params)
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
