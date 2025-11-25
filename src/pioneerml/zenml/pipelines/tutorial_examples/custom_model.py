"""
Custom model pipeline example for tutorials.

This pipeline demonstrates how to create and use custom models
in ZenML pipelines.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from zenml import pipeline, step

from pioneerml.models.base import GraphModel
from pioneerml.training import GraphDataModule, GraphLightningModule
from pioneerml.zenml.utils import detect_available_accelerator


class SimpleGCN(GraphModel):
    """A simple custom GCN model for tutorials."""

    def __init__(self, num_classes: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.conv1 = GCNConv(5, hidden_dim)  # 5 input features
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch):
        x, edge_index, batch_indices = batch.x, batch.edge_index, batch.batch

        # Graph convolution layers
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()

        # Global pooling
        x = global_mean_pool(x, batch_indices)

        # Classification
        return self.classifier(x)


def create_simple_synthetic_data(num_samples: int = 100) -> list[Data]:
    """Create synthetic graph data for the custom model."""
    data = []
    for _ in range(num_samples):
        # Create graphs with 4-8 nodes
        num_nodes = torch.randint(4, 9, (1,)).item()
        x = torch.randn(num_nodes, 5)
        # Create random edges
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        edge_attr = torch.randn(edge_index.shape[1], 4)

        # Random class label
        label = torch.randint(0, 3, (1,)).item()
        y = torch.zeros(3)
        y[label] = 1.0

        data.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    return data


@step
def create_custom_data() -> list[Data]:
    """Step to create custom synthetic data."""
    return create_simple_synthetic_data(150)


@step
def create_custom_datamodule(data: list[Data]) -> GraphDataModule:
    """Step to create DataModule."""
    return GraphDataModule(dataset=data, val_split=0.3, batch_size=16)


@step
def create_custom_model(num_classes: int = 3, hidden_dim: int = 64) -> SimpleGCN:
    """Step to create custom model."""
    return SimpleGCN(num_classes=num_classes, hidden_dim=hidden_dim)


@step
def create_custom_lightning_module(model: SimpleGCN) -> GraphLightningModule:
    """Step to create Lightning module with custom model."""
    return GraphLightningModule(model, task="classification", lr=5e-4)


@step
def train_custom_model(
    lightning_module: GraphLightningModule,
    datamodule: GraphDataModule
) -> GraphLightningModule:
    """Step to train the custom model."""
    import pytorch_lightning as pl

    accelerator, devices = detect_available_accelerator()

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=5,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    trainer.fit(lightning_module, datamodule=datamodule)
    return lightning_module.eval()


@pipeline
def custom_model_pipeline():
    """Custom model training pipeline example."""
    data = create_custom_data()
    datamodule = create_custom_datamodule(data)
    model = create_custom_model()
    lightning_module = create_custom_lightning_module(model)
    trained_module = train_custom_model(lightning_module, datamodule)
    return trained_module, datamodule
