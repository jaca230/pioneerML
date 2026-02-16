import pytorch_lightning as pl
import torch.nn as nn
from zenml import step

from pioneerml.common.loader import GroupClassifierGraphLoader
from pioneerml.common.models.classifiers import GroupClassifier
from pioneerml.common.pipeline_utils.misc import LightningWarningFilter
from pioneerml.common.training import (
    GraphLightningModule,
    build_early_stopping_callback,
    maybe_compile_model,
    restore_eager_model_if_compiled,
)
from pioneerml.common.zenml.utils import detect_available_accelerator
from pioneerml.pipelines.training.group_classification.steps.config import resolve_step_config

_WARNING_FILTER = LightningWarningFilter()


def _apply_lightning_warnings_filter() -> None:
    _WARNING_FILTER.apply_default()


def _merge_config(base: dict, override) -> dict:
    merged = dict(base)
    if override is not None:
        merged.update({k: v for k, v in override.items() if v is not None})
    return merged


def _as_optional_split(value) -> str | None:
    if value in (None, "", "none", "None"):
        return None
    return str(value).strip().lower()


def _resolve_stage_loader_config(
    cfg: dict,
    *,
    stage: str,
    forced_batch_size: int | None = None,
) -> dict:
    raw = cfg.get("loader_config")
    base_cfg: dict = {}
    stage_cfg: dict = {}
    if isinstance(raw, dict):
        if any(k in raw for k in ("base", "train", "val")):
            base_cfg = dict(raw.get("base") or {})
            stage_cfg = dict(raw.get(stage) or {})
        else:
            base_cfg = dict(raw)
    merged = {**base_cfg, **stage_cfg}

    if forced_batch_size is not None:
        merged["batch_size"] = int(forced_batch_size)
    else:
        merged.setdefault("batch_size", int(cfg.get("batch_size", 64)))
    merged.setdefault("chunk_row_groups", int(cfg.get("chunk_row_groups", 4)))
    merged.setdefault("chunk_workers", int(cfg.get("chunk_workers", 0)))
    merged.setdefault("mode", "train")
    return merged


def _build_stage_loader(
    *,
    parquet_paths: list[str],
    loader_cfg: dict,
) -> GroupClassifierGraphLoader:
    return GroupClassifierGraphLoader(
        parquet_paths=[str(p) for p in parquet_paths],
        mode=str(loader_cfg.get("mode", "train")),
        batch_size=max(1, int(loader_cfg.get("batch_size", 64))),
        row_groups_per_chunk=max(
            1,
            int(loader_cfg.get("chunk_row_groups", loader_cfg.get("row_groups_per_chunk", 4))),
        ),
        num_workers=max(0, int(loader_cfg.get("chunk_workers", loader_cfg.get("num_workers", 0)))),
        split=_as_optional_split(loader_cfg.get("split")),
        train_fraction=float(loader_cfg.get("train_fraction", 0.9)),
        val_fraction=float(loader_cfg.get("val_fraction", 0.05)),
        test_fraction=float(loader_cfg.get("test_fraction", 0.05)),
        split_seed=int(loader_cfg.get("split_seed", 0)),
        sample_fraction=(
            None
            if loader_cfg.get("sample_fraction") in (None, "", "none", "None")
            else float(loader_cfg.get("sample_fraction"))
        ),
    )


def _fit_with_loaders(
    *,
    module: GraphLightningModule,
    train_loader,
    val_loader,
    max_epochs: int,
    grad_clip: float | None,
    trainer_kwargs: dict | None,
    early_stopping_cfg: dict | None,
) -> GraphLightningModule:
    merged_trainer_kwargs = dict(trainer_kwargs or {})
    if grad_clip is not None:
        merged_trainer_kwargs.setdefault("gradient_clip_val", float(grad_clip))
    callbacks = list(merged_trainer_kwargs.get("callbacks") or [])
    es_callback = build_early_stopping_callback(early_stopping_cfg)
    if es_callback is not None:
        callbacks.append(es_callback)
    if callbacks:
        merged_trainer_kwargs["callbacks"] = callbacks
    accelerator, devices = detect_available_accelerator()
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=int(max_epochs),
        enable_checkpointing=False,
        logger=False,
        **merged_trainer_kwargs,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    restore_eager_model_if_compiled(module)
    return module


@step
def train_group_classifier(
    parquet_paths: list[str],
    pipeline_config: dict | None = None,
    hpo_params: dict | None = None,
) -> GraphLightningModule:
    _apply_lightning_warnings_filter()

    defaults = {
        "max_epochs": 10,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "grad_clip": 2.0,
        "scheduler_step_size": 2,
        "scheduler_gamma": 0.5,
        "threshold": 0.5,
        "trainer_kwargs": {"enable_progress_bar": True},
        "batch_size": 64,
        "shuffle": True,
        "chunk_row_groups": 4,
        "chunk_workers": 0,
        "early_stopping": {
            "enabled": False,
            "monitor": "val_loss",
            "mode": "min",
            "patience": 5,
            "min_delta": 0.0,
            "min_delta_mode": "absolute",
        },
        "compile": {"enabled": False, "mode": "default"},
        "model": {
            "in_dim": 4,
            "edge_dim": 4,
            "hidden": 200,
            "heads": 4,
            "num_blocks": 2,
            "dropout": 0.1,
        },
    }
    cfg = _merge_config(defaults, resolve_step_config(pipeline_config, "train"))
    if hpo_params:
        cfg = _merge_config(cfg, hpo_params)

    model_cfg = dict(cfg.get("model") or {})
    if "in_dim" not in model_cfg:
        model_cfg["in_dim"] = int(GroupClassifierGraphLoader.NODE_FEATURE_DIM)
    if "edge_dim" not in model_cfg:
        model_cfg["edge_dim"] = int(GroupClassifierGraphLoader.EDGE_FEATURE_DIM)

    hidden = int(model_cfg.get("hidden", 200))
    heads = int(model_cfg.get("heads", 4))
    if hidden % heads != 0:
        raise ValueError(f"hidden ({hidden}) must be divisible by heads ({heads})")

    model = GroupClassifier(
        in_dim=int(model_cfg["in_dim"]),
        edge_dim=int(model_cfg["edge_dim"]),
        hidden=hidden,
        heads=heads,
        num_blocks=int(model_cfg.get("num_blocks", 2)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        num_classes=int(GroupClassifierGraphLoader.NUM_CLASSES),
    )
    model = maybe_compile_model(model, cfg.get("compile"), context="train_group_classifier")

    module = GraphLightningModule(
        model,
        task="classification",
        loss_fn=nn.BCEWithLogitsLoss(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
        threshold=float(cfg.get("threshold", 0.5)),
        scheduler_step_size=int(cfg["scheduler_step_size"]) if cfg.get("scheduler_step_size") is not None else None,
        scheduler_gamma=float(cfg["scheduler_gamma"]),
    )

    train_loader_cfg = _resolve_stage_loader_config(cfg, stage="train")
    val_loader_cfg = _resolve_stage_loader_config(cfg, stage="val")
    train_loader_provider = _build_stage_loader(parquet_paths=parquet_paths, loader_cfg=train_loader_cfg)
    val_loader_provider = _build_stage_loader(parquet_paths=parquet_paths, loader_cfg=val_loader_cfg)
    if not train_loader_provider.include_targets or not val_loader_provider.include_targets:
        raise RuntimeError("GroupClassifierGraphLoader must run in train mode for training/validation.")

    train_loader = train_loader_provider.make_dataloader(shuffle_batches=bool(cfg.get("shuffle", True)))
    val_loader = val_loader_provider.make_dataloader(shuffle_batches=False)

    return _fit_with_loaders(
        module=module,
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=int(cfg["max_epochs"]),
        grad_clip=float(cfg["grad_clip"]) if cfg.get("grad_clip") is not None else None,
        trainer_kwargs=dict(cfg.get("trainer_kwargs") or {}),
        early_stopping_cfg=dict(cfg.get("early_stopping") or {}),
    )
