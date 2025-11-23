"""
Training utilities and loops for PIONEER models.

This module will contain:
- Training loops (basic PyTorch and PyTorch Lightning)
- Loss functions
- Optimizers and schedulers
- Callbacks and hooks
- Checkpointing utilities
"""

# Placeholder for future training utilities
# Training functions currently reside in notebooks
# TODO: Extract and modularize training code from notebooks

from pioneerml.training.lightning import GraphLightningModule
from pioneerml.training.datamodules import (
    GraphDataModule,
    GroupClassificationDataModule,
    SplitterDataModule,
    PionStopDataModule,
)
from pioneerml.training.visualization import plot_loss_curves

__all__ = [
    "GraphLightningModule",
    "GraphDataModule",
    "GroupClassificationDataModule",
    "SplitterDataModule",
    "PionStopDataModule",
    "plot_loss_curves",
]
