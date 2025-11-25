"""
Basic training pipeline example for tutorials.

This pipeline demonstrates the fundamental ZenML pipeline structure
for training a simple model.
"""

import torch
from torch_geometric.data import Data
from zenml import pipeline, step

from pioneerml.models import GroupClassifier
from pioneerml.training import GraphDataModule, GraphLightningModule
from pioneerml.zenml.utils import detect_available_accelerator
from pioneerml.zenml.materializers import (
    GraphDataModuleMaterializer,
    PyGDataListMaterializer,
)


def create_simple_synthetic_data(num_samples: int = 100) -> list[Data]:
    """Create clustered synthetic graphs that are easy to learn."""
    class_offsets = torch.tensor(
        [
            [2.0, 0.0, 0.5, 0.0, 0.0],   # pi cluster: high first feature
            [0.0, 2.0, 0.0, 0.5, 0.0],   # mu cluster: high second feature
            [-2.0, -2.0, -0.5, 0.0, 0.5],  # e+ cluster: negative drift
        ]
    )
    data: list[Data] = []
    for _ in range(num_samples):
        num_nodes = torch.randint(6, 10, (1,)).item()
        label = torch.randint(0, 3, (1,)).item()

        # Clustered node features with modest noise so the classifier can separate classes.
        x = torch.randn(num_nodes, 5) * 0.4 + class_offsets[label]

        # Dense-ish random edges; attributes keep small noise.
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 3))
        edge_attr = torch.randn(edge_index.shape[1], 4) * 0.3

        y = torch.zeros(3)
        y[label] = 1.0
        data.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    return data


@step(output_materializers=PyGDataListMaterializer, enable_cache=False)
def create_data() -> list[Data]:
    """Step to create synthetic training data."""
    return create_simple_synthetic_data(200)


@step(output_materializers=GraphDataModuleMaterializer, enable_cache=False)
def create_datamodule(data: list[Data]) -> GraphDataModule:
    """Step to create a DataModule from the data."""
    # Use CPU-friendly defaults; keep workers=0 to avoid multiprocessing issues in constrained envs.
    return GraphDataModule(dataset=data, val_split=0.2, batch_size=32, num_workers=0)


@step
def create_model(num_classes: int = 3) -> GroupClassifier:
    """Step to create the model."""
    return GroupClassifier(num_classes=num_classes, hidden=64, num_blocks=1)


@step
def create_lightning_module(model: GroupClassifier) -> GraphLightningModule:
    """Step to create the Lightning module."""
    return GraphLightningModule(model, task="classification", lr=1e-3)


@step
def train_model(
    lightning_module: GraphLightningModule,
    datamodule: GraphDataModule
) -> GraphLightningModule:
    """Step to train the model."""
    import pytorch_lightning as pl

    accelerator, devices = detect_available_accelerator()

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=3,  # Quick training for tutorial
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,  # Quiet for tutorial
    )

    trainer.fit(lightning_module, datamodule=datamodule)
    return lightning_module.eval()


@pipeline
def basic_training_pipeline():
    """Basic training pipeline example."""
    data = create_data()
    datamodule = create_datamodule(data)
    model = create_model()
    lightning_module = create_lightning_module(model)
    trained_module = train_model(lightning_module, datamodule)
    return trained_module, datamodule
