"""
Lightweight PyTorch Lightning utilities for graph models.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Batch

from pioneerml.models.base import GraphModel


class GraphLightningModule(pl.LightningModule):
    """
    Generic LightningModule wrapper for GraphModel instances.

    Handles common training/validation loops for classification and regression
    tasks while delegating the forward pass to the underlying graph model.
    """

    def __init__(
        self,
        model: GraphModel,
        *,
        task: str = "classification",
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer_cls: type[optim.Optimizer] = optim.AdamW,
    ):
        super().__init__()
        self.model = model
        self.task = task
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_cls = optimizer_cls

        if loss_fn is not None:
            self.loss_fn = loss_fn
        elif task == "classification":
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.MSELoss()

    def forward(self, batch: Batch) -> torch.Tensor:
        return self.model(batch)

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        preds, target = self._shared_step(batch)
        loss = self.loss_fn(preds, target)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        metrics = self._compute_metrics(preds, target, prefix="train")
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        preds, target = self._shared_step(batch)
        loss = self.loss_fn(preds, target)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        metrics = self._compute_metrics(preds, target, prefix="val")
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False)

    def predict_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        return self(batch)

    def configure_optimizers(self):
        return self.optimizer_cls(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _shared_step(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(batch, "y"):
            raise AttributeError("Batch is missing target attribute 'y' required for training.")
        preds = self(batch)
        target = batch.y

        # Ensure target shape matches preds for BCE/CE losses
        if target.dim() == 1 and preds.dim() == 2 and target.numel() % preds.shape[-1] == 0:
            target = target.view(-1, preds.shape[-1])
        return preds, target

    def _compute_metrics(self, preds: torch.Tensor, target: torch.Tensor, prefix: str) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        if self.task == "classification":
            probs = torch.sigmoid(preds)
            preds_binary = (probs > 0.5).float()
            metrics[f"{prefix}_accuracy"] = (preds_binary == target.float()).float().mean().item()
        else:
            metrics[f"{prefix}_mae"] = torch.abs(preds - target).mean().item()

        return metrics
