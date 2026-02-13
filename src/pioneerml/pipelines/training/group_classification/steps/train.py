import pytorch_lightning as pl
import torch.nn as nn
from zenml import step

from pioneerml.common.loader import GroupClassifierGraphLoader
from pioneerml.common.models.classifiers import GroupClassifier
from pioneerml.common.pipeline_utils.misc import LightningWarningFilter
from pioneerml.common.training.lightning import GraphLightningModule
from pioneerml.common.zenml.utils import detect_available_accelerator
from pioneerml.pipelines.training.group_classification.dataset import GroupClassifierDataset
from pioneerml.pipelines.training.group_classification.steps.config import resolve_step_config

_WARNING_FILTER = LightningWarningFilter()


def _apply_lightning_warnings_filter() -> None:
    _WARNING_FILTER.apply_default()


def _merge_config(base: dict, override) -> dict:
    merged = dict(base)
    if override is not None:
        merged.update({k: v for k, v in override.items() if v is not None})
    return merged


def _fit_with_loaders(
    *,
    module: GraphLightningModule,
    train_loader,
    val_loader,
    max_epochs: int,
    grad_clip: float | None,
    trainer_kwargs: dict | None,
) -> GraphLightningModule:
    merged_trainer_kwargs = dict(trainer_kwargs or {})
    if grad_clip is not None:
        merged_trainer_kwargs.setdefault("gradient_clip_val", float(grad_clip))
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
    return module


@step
def train_group_classifier(
    dataset: GroupClassifierDataset,
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
        model_cfg["in_dim"] = int(dataset.data.x.shape[-1])
    if "edge_dim" not in model_cfg:
        model_cfg["edge_dim"] = int(dataset.data.edge_attr.shape[-1])

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
        num_classes=int(dataset.targets.shape[-1]),
    )

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

    base_loader = getattr(dataset, "loader", None)
    if not isinstance(base_loader, GroupClassifierGraphLoader):
        raise RuntimeError("Dataset is missing GroupClassifierGraphLoader required for chunked training.")
    if not base_loader.include_targets:
        raise RuntimeError("GroupClassifierGraphLoader must run in train mode for training.")

    batch_size = int(cfg.get("batch_size", 64))
    chunk_row_groups = int(cfg.get("chunk_row_groups", 4))
    chunk_workers = int(cfg.get("chunk_workers", 0))

    loader_provider = base_loader.with_runtime(
        batch_size=batch_size,
        row_groups_per_chunk=chunk_row_groups,
        num_workers=chunk_workers,
    )
    train_loader = loader_provider.make_dataloader(shuffle_batches=bool(cfg.get("shuffle", True)))
    val_loader = loader_provider.make_dataloader(shuffle_batches=False)

    return _fit_with_loaders(
        module=module,
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=int(cfg["max_epochs"]),
        grad_clip=float(cfg["grad_clip"]) if cfg.get("grad_clip") is not None else None,
        trainer_kwargs=dict(cfg.get("trainer_kwargs") or {}),
    )
