import warnings
from typing import Mapping

import pytorch_lightning as pl
import torch
import torch.nn as nn
from zenml import step

from pioneerml.models.classifiers import GroupClassifier
from pioneerml.training.lightning import GraphLightningModule
from pioneerml.zenml.utils import detect_available_accelerator

from pioneerml.zenml.pipelines.training.group_classification.batch import GroupClassifierBatch


def _apply_lightning_warnings_filter() -> None:
    warnings.filterwarnings(
        "ignore",
        message="The 'train_dataloader' does not have many workers.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="isinstance\\(treespec, LeafSpec\\) is deprecated.*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="isinstance\\(treespec, LeafSpec\\) is deprecated.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="isinstance\\(treespec, LeafSpec\\) is deprecated.*",
        category=Warning,
        module="pytorch_lightning\\.utilities\\._pytree",
    )
    warnings.filterwarnings(
        "ignore",
        message="The 'val_dataloader' does not have many workers.*",
        category=UserWarning,
    )


def _merge_config(
    base: dict,
    override: Mapping | None,
) -> dict:
    if override is None:
        return base
    merged = dict(base)
    merged.update({k: v for k, v in override.items() if v is not None})
    return merged


@step
def train_group_classifier(
    batch: GroupClassifierBatch,
    *,
    step_config: dict | None = None,
    hpo_params: dict | None = None,
) -> GraphLightningModule:
    _apply_lightning_warnings_filter()

    defaults = {
        "max_epochs": 10,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "trainer_kwargs": {},
        "model": {
            "in_dim": 4,
            "edge_dim": 4,
            "hidden": 200,
            "heads": 4,
            "num_blocks": 2,
            "dropout": 0.1,
        },
    }
    cfg = _merge_config(defaults, step_config)
    if hpo_params is not None:
        cfg = _merge_config(cfg, hpo_params)

    model_cfg = dict(cfg.get("model") or {})
    if "in_dim" not in model_cfg:
        model_cfg["in_dim"] = int(batch.data.x.shape[-1])
    if "edge_dim" not in model_cfg:
        model_cfg["edge_dim"] = int(batch.data.edge_attr.shape[-1])
    model = GroupClassifier(
        in_dim=int(model_cfg["in_dim"]),
        edge_dim=int(model_cfg["edge_dim"]),
        hidden=int(model_cfg["hidden"]),
        heads=int(model_cfg["heads"]),
        num_blocks=int(model_cfg["num_blocks"]),
        dropout=float(model_cfg["dropout"]),
        num_classes=int(batch.targets.shape[-1]),
    )
    module = GraphLightningModule(
        model,
        task="classification",
        loss_fn=nn.BCEWithLogitsLoss(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    accelerator, devices = detect_available_accelerator()
    trainer = torch.utils.data.DataLoader(
        [batch.data],
        batch_size=1,
        shuffle=False,
        collate_fn=lambda items: items[0],
    )

    trainer_kwargs = dict(cfg.get("trainer_kwargs") or {})
    lightning_trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=int(cfg["max_epochs"]),
        enable_checkpointing=False,
        logger=False,
        **trainer_kwargs,
    )
    lightning_trainer.fit(module, train_dataloaders=trainer, val_dataloaders=trainer)
    return module
