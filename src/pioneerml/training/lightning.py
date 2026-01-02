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

        # Simple histories for plotting later
        self.train_loss_history: list[float] = []
        self.val_loss_history: list[float] = []
        self.train_epoch_loss_history: list[float] = []
        self.val_epoch_loss_history: list[float] = []
        self._train_loss_sum: float = 0.0
        self._train_loss_count: int = 0
        self._val_loss_sum: float = 0.0
        self._val_loss_count: int = 0

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
        bs = self._get_batch_size(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)

        metrics = self._compute_metrics(preds, target, prefix="train")
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, batch_size=bs)
        self.train_loss_history.append(loss.detach().cpu().item())
        self._train_loss_sum += loss.detach().cpu().item() * bs
        self._train_loss_count += bs
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        preds, target = self._shared_step(batch)
        loss = self.loss_fn(preds, target)
        bs = self._get_batch_size(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)

        metrics = self._compute_metrics(preds, target, prefix="val")
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, batch_size=bs)
        self.val_loss_history.append(loss.detach().cpu().item())
        self._val_loss_sum += loss.detach().cpu().item() * bs
        self._val_loss_count += bs

    def on_train_epoch_end(self) -> None:
        if self._train_loss_count > 0:
            self.train_epoch_loss_history.append(self._train_loss_sum / self._train_loss_count)
        self._train_loss_sum = 0.0
        self._train_loss_count = 0

    def on_validation_epoch_end(self) -> None:
        if self._val_loss_count > 0:
            self.val_epoch_loss_history.append(self._val_loss_sum / self._val_loss_count)
        self._val_loss_sum = 0.0
        self._val_loss_count = 0

    def predict_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        return self(batch)

    def configure_optimizers(self):
        return self.optimizer_cls(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _shared_step(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(batch, "y"):
            raise AttributeError("Batch is missing target attribute 'y' required for training.")
        raw_preds = self(batch)
        # If the model returns multiple outputs, use the first for loss/metrics
        preds = raw_preds[0] if isinstance(raw_preds, (tuple, list)) else raw_preds
        target = batch.y

        # Ensure target shape matches preds for BCE/CE losses
        if target.dim() == 1 and preds.dim() == 2 and target.numel() % preds.shape[-1] == 0:
            target = target.view(-1, preds.shape[-1])
        return preds, target

    @staticmethod
    def _get_batch_size(batch: Batch) -> int:
        if hasattr(batch, "num_graphs"):
            return int(batch.num_graphs)
        if hasattr(batch, "batch"):
            return int(batch.batch.max().item() + 1)
        return 1

    def _compute_metrics(self, preds: torch.Tensor, target: torch.Tensor, prefix: str) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        if self.task == "classification":
            probs = torch.sigmoid(preds)
            preds_binary = (probs > 0.5).float()
            metrics[f"{prefix}_accuracy"] = (preds_binary == target.float()).float().mean().item()
        else:
            metrics[f"{prefix}_mae"] = torch.abs(preds - target).mean().item()

        return metrics
