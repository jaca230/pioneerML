from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from zenml import pipeline, step

from pioneerml.training.lightning import GraphLightningModule
from pioneerml.training.datamodules import GroupClassificationDataModule
from pioneerml.zenml.utils import detect_available_accelerator
from .datamodule_factory import build_datamodule
from .model_factory import build_model


@step
def build_group_classification_datamodule_step(**kwargs) -> GroupClassificationDataModule:
    return build_datamodule(**kwargs)


@step(enable_cache=False)
def run_group_classification_hparam_search(
    datamodule: GroupClassificationDataModule,
    *,
    max_epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Mapping[str, Any]:
    # Simple single-run config (Optuna can be re-added on top later)
    return {"max_epochs": max_epochs, "lr": lr, "weight_decay": weight_decay}


@step
def train_best_group_classifier(
    hparams: Mapping[str, Any],
    datamodule: GroupClassificationDataModule,
) -> GraphLightningModule:
    model = build_model(num_classes=datamodule.train_dataset.dataset.num_classes if datamodule.train_dataset else 3)
    module = GraphLightningModule(
        model,
        task="classification",
        loss_fn=nn.BCEWithLogitsLoss(),
        lr=hparams.get("lr", 1e-3),
        weight_decay=hparams.get("weight_decay", 1e-4),
    )

    accelerator, devices = detect_available_accelerator()
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=int(hparams.get("max_epochs", 20)),
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(module, datamodule=datamodule)
    return module


@step
def collect_group_classification_predictions(
    module: GraphLightningModule,
    datamodule: GroupClassificationDataModule,
) -> Tuple[torch.Tensor, torch.Tensor]:
    module.eval()
    accelerator, devices = detect_available_accelerator()
    device = torch.device("cuda" if accelerator == "gpu" and torch.cuda.is_available() else "cpu")
    module.to(device)

    preds = []
    targets = []
    for batch in datamodule.val_dataloader() or datamodule.train_dataloader():
        batch = batch.to(device)
        with torch.no_grad():
            out = module(batch)
        preds.append(out.cpu())
        targets.append(batch.y.cpu())
    return torch.cat(preds), torch.cat(targets)


@pipeline
def group_classification_pipeline(
    datamodule_params: Optional[Mapping[str, Any]] = None,
    train_params: Optional[Mapping[str, Any]] = None,
):
    dm_kwargs = dict(datamodule_params or {})
    train_kwargs = dict(train_params or {})

    dm = build_group_classification_datamodule_step.with_options(parameters=dm_kwargs)()
    hparams = run_group_classification_hparam_search(dm)
    module = train_best_group_classifier.with_options(parameters=train_kwargs)(hparams, dm)
    preds, targets = collect_group_classification_predictions(module, dm)
    return module, dm, preds, targets, hparams
