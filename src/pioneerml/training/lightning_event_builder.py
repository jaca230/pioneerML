"""
PyTorch LightningModule wrapper for EventBuilder affinity training.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import pytorch_lightning as pl

from pioneerml.models.components.event_builder import EventBuilder


class EventBuilderLightning(pl.LightningModule):
    """
    LightningModule for training EventBuilder on affinity targets.
    """

    def __init__(
        self,
        model: EventBuilder,
        *,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = nn.BCELoss(reduction="none")  # model outputs probabilities

    def forward(self, batch):
        x, edge_index, edge_attr, group_indices, batch_indices = batch
        return self.model(x, edge_index, edge_attr, group_indices, batch_indices)

    def _compute_loss(self, preds, targets, batch_indices):
        batch_ids = batch_indices.unsqueeze(1)
        event_mask = (batch_ids == batch_ids.T).float()
        raw_loss = self.loss_fn(preds, targets)
        loss = (raw_loss * event_mask).sum() / (event_mask.sum() + 1e-6)
        return loss, event_mask

    def training_step(self, batch, batch_idx):
        x, edge_index, edge_attr, group_indices, batch_indices, targets = batch
        preds = self.model(x, edge_index, edge_attr, group_indices, batch_indices)
        loss, event_mask = self._compute_loss(preds, targets, batch_indices)

        # positive-class accuracy within event blocks
        with torch.no_grad():
            pos_mask = (targets == 1) & (event_mask == 1)
            acc = (preds[pos_mask] > 0.5).float().mean().item() if pos_mask.any() else 0.0
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_indices.size(0))
        self.log("train_pos_acc", acc, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_indices.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x, edge_index, edge_attr, group_indices, batch_indices, targets = batch
        preds = self.model(x, edge_index, edge_attr, group_indices, batch_indices)
        loss, event_mask = self._compute_loss(preds, targets, batch_indices)
        with torch.no_grad():
            pos_mask = (targets == 1) & (event_mask == 1)
            acc = (preds[pos_mask] > 0.5).float().mean().item() if pos_mask.any() else 0.0
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_indices.size(0))
        self.log("val_pos_acc", acc, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_indices.size(0))
        return {"preds": preds.detach(), "targets": targets.detach(), "event_mask": event_mask.detach()}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

