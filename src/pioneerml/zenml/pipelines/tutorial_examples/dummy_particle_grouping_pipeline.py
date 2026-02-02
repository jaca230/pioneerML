"""
Dummy particle grouping ZenML pipeline.

This pipeline mirrors the synthetic grouping logic used in the legacy
notebook: we generate simple per-hit features, assign three binary labels
(energy, hit count, spatial spread), and train a GroupClassifier on the
resulting multi-label task.
"""

import numpy as np
import torch
import pytorch_lightning as pl
from zenml import pipeline, step

from pioneerml.data.datasets.graph_group import GraphRecord
from pioneerml.models.classifiers.group_classifier import GroupClassifier
from pioneerml.training.datamodules import GroupClassificationDataModule
from pioneerml.training.lightning import GraphLightningModule
from pioneerml.zenml.materializers import TorchTensorMaterializer
from pioneerml.zenml.utils import detect_available_accelerator


def _make_dummy_record(
    rng: np.random.Generator,
    event_id: int,
    *,
    thresholds: dict[str, float],
    num_hits_range: tuple[int, int],
) -> GraphRecord:
    num_hits = int(rng.integers(low=num_hits_range[0], high=num_hits_range[1] + 1))

    coord = rng.normal(size=num_hits).astype(np.float32)
    z_pos = rng.normal(size=num_hits).astype(np.float32)
    energy = np.abs(rng.normal(size=num_hits)).astype(np.float32)
    view = rng.integers(0, 2, size=num_hits).astype(np.float32)

    energy_mean = float(energy.mean())
    spatial_spread = float(coord.std())

    labels: list[int] = []
    labels.append(0 if energy_mean > thresholds["energy"] else 1)
    labels.append(2 if num_hits > thresholds["hits"] else 3)
    labels.append(4 if spatial_spread > thresholds["spread"] else 5)

    return GraphRecord(
        coord=coord,
        z=z_pos,
        energy=energy,
        view=view,
        labels=labels,
        event_id=event_id,
        group_id=event_id,
    )


@step
def build_dummy_datamodule(
    num_samples: int = 1024,
    *,
    num_classes: int = 6,
    num_hits_range: tuple[int, int] = (6, 40),
    thresholds: dict[str, float] | None = None,
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
) -> GroupClassificationDataModule:
    rng = np.random.default_rng(seed)
    thresholds = thresholds or {"energy": 1.0, "hits": 20.0, "spread": 1.0}

    records = [
        _make_dummy_record(rng, event_id=i, thresholds=thresholds, num_hits_range=num_hits_range)
        for i in range(num_samples)
    ]

    dm = GroupClassificationDataModule(
        records,
        num_classes=num_classes,
        batch_size=batch_size,
        val_split=val_split,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
    )
    dm.setup(stage="fit")
    return dm


@step
def build_dummy_module(
    *,
    num_classes: int = 6,
    hidden: int = 192,
    num_blocks: int = 3,
    dropout: float = 0.1,
    heads: int = 4,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
) -> GraphLightningModule:
    model = GroupClassifier(
        in_dim=4,
        edge_dim=4,
        hidden=hidden,
        heads=heads,
        num_blocks=num_blocks,
        dropout=dropout,
        num_classes=num_classes,
    )
    return GraphLightningModule(model=model, task="classification", lr=lr, weight_decay=weight_decay)


@step
def train_dummy_module(
    module: GraphLightningModule,
    datamodule: GroupClassificationDataModule,
    *,
    max_epochs: int = 20,
    limit_train_batches: int | float | None = None,
    limit_val_batches: int | float | None = None,
) -> GraphLightningModule:
    accelerator, devices = detect_available_accelerator()
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, datamodule=datamodule)
    return module.eval()


@step(
    enable_cache=False,
    output_materializers=(TorchTensorMaterializer, TorchTensorMaterializer),
)
def collect_dummy_predictions(
    module: GraphLightningModule, datamodule: GroupClassificationDataModule
) -> tuple[torch.Tensor, torch.Tensor]:
    datamodule.setup(stage="fit")
    val_loader = datamodule.val_dataloader()
    if isinstance(val_loader, list) and len(val_loader) == 0:
        val_loader = datamodule.train_dataloader()

    preds, targets = [], []
    device = next(module.parameters()).device
    module.eval()
    for batch in val_loader:
        batch = batch.to(device)
        with torch.no_grad():
            out = module(batch).detach().cpu()
        target = batch.y.detach().cpu()

        if target.dim() == 1 and out.dim() == 2 and target.numel() % out.shape[-1] == 0:
            target = target.view(-1, out.shape[-1])
        elif target.dim() == 1:
            target = target.unsqueeze(0)

        preds.append(out)
        targets.append(target)

    return torch.cat(preds), torch.cat(targets)


@pipeline
def dummy_particle_grouping_pipeline():
    datamodule = build_dummy_datamodule()
    module = build_dummy_module()
    trained_module = train_dummy_module(module, datamodule)
    predictions, targets = collect_dummy_predictions(trained_module, datamodule)
    return trained_module, datamodule, predictions, targets
