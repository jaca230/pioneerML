"""
ZenML training pipeline for PIONEER ML models.

This pipeline builds synthetic data, trains a GroupClassifier with Lightning,
and logs metrics.
"""

import torch
from torch_geometric.data import Data
from zenml import pipeline, step

from pioneerml.models import GroupClassifier
from pioneerml.training import GraphDataModule, GraphLightningModule
from pioneerml.zenml.materializers import TorchTensorMaterializer


def make_synthetic_group(num_nodes: int, num_classes: int) -> Data:
    class_offsets = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )
    label = torch.randint(0, num_classes, (1,)).item()
    x = torch.randn(num_nodes, 5) * 1.2 + class_offsets[label]
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
    edge_attr = torch.randn(edge_index.shape[1], 4)
    y = torch.zeros(num_classes)
    y[label] = 1.0
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


@step
def build_datamodule(
    num_samples: int = 256,
    num_nodes: int = 16,
    num_classes: int = 3,
    batch_size: int = 16,
    val_split: float = 0.25,
) -> GraphDataModule:
    records = [make_synthetic_group(num_nodes, num_classes) for _ in range(num_samples)]
    return GraphDataModule(dataset=records, val_split=val_split, batch_size=batch_size)


@step
def build_module(num_classes: int = 3, lr: float = 5e-4) -> GraphLightningModule:
    model = GroupClassifier(num_classes=num_classes)
    return GraphLightningModule(model, task="classification", lr=lr)


@step
def train_module(module: GraphLightningModule, datamodule: GraphDataModule) -> GraphLightningModule:
    import pytorch_lightning as pl

    # Import hardware detection utility
    from pioneerml.zenml.utils import detect_available_accelerator

    # Auto-detect available hardware
    accelerator, devices = detect_available_accelerator()

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=5,
        limit_train_batches=5,
        limit_val_batches=1,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(module, datamodule=datamodule)
    return module.eval()


@step(
    enable_cache=False,
    output_materializers=(TorchTensorMaterializer, TorchTensorMaterializer)
)
def collect_predictions(module: GraphLightningModule, datamodule: GraphDataModule) -> tuple[torch.Tensor, torch.Tensor]:
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
            preds.append(module(batch).detach().cpu())
            targets.append(batch.y.detach().cpu())

    return torch.cat(preds), torch.cat(targets)


@pipeline
def zenml_training_pipeline():
    dm = build_datamodule()
    module = build_module()
    trained_module = train_module(module, dm)
    predictions, targets = collect_predictions(trained_module, dm)
    return trained_module, dm, predictions, targets
