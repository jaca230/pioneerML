"""
Lightweight PyTorch Lightning utilities for graph models.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional
import warnings

warnings.filterwarnings(
    "ignore",
    message="isinstance\\(treespec, LeafSpec\\) is deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    message="`isinstance\\(treespec, LeafSpec\\)` is deprecated.*",
    category=FutureWarning,
)

import pytorch_lightning as pl
try:  # pragma: no cover - optional import path
    from lightning_fabric.utilities.rank_zero import LightningDeprecationWarning
except Exception:  # pragma: no cover
    LightningDeprecationWarning = None
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Batch

from pioneerml.common.models.base import GraphModel

if LightningDeprecationWarning is not None:
    warnings.filterwarnings("ignore", category=LightningDeprecationWarning)


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
        threshold: float = 0.5,
        scheduler_step_size: Optional[int] = None,
        scheduler_gamma: float = 0.5,
    ):
        super().__init__()
        self.model = model
        self.task = task
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_cls = optimizer_cls
        self.threshold = threshold
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma

        # Simple histories for plotting later
        self.train_loss_history: list[float] = []
        self.val_loss_history: list[float] = []
        self.train_epoch_loss_history: list[float] = []
        self.val_epoch_loss_history: list[float] = []
        self._train_loss_sum: float = 0.0
        self._train_loss_count: int = 0
        self._val_loss_sum: float = 0.0
        self._val_loss_count: int = 0
        self._train_confusion: Optional[torch.Tensor] = None
        self._val_confusion: Optional[torch.Tensor] = None

        if loss_fn is not None:
            self.loss_fn = loss_fn
        elif task == "classification":
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.MSELoss()

    def forward(self, batch: Batch) -> torch.Tensor:
        return self._model_forward(batch)

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        preds, target = self._shared_step(batch)
        loss = self.loss_fn(preds, target)
        bs = self._get_batch_size(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)

        metrics = self._compute_metrics(preds, target, prefix="train")
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, batch_size=bs)
        self._update_confusion(preds, target, is_train=True)
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
        self._update_confusion(preds, target, is_train=False)
        self.val_loss_history.append(loss.detach().cpu().item())
        self._val_loss_sum += loss.detach().cpu().item() * bs
        self._val_loss_count += bs

    def on_train_epoch_end(self) -> None:
        if self._train_loss_count > 0:
            self.train_epoch_loss_history.append(self._train_loss_sum / self._train_loss_count)
        self._train_loss_sum = 0.0
        self._train_loss_count = 0
        self._log_confusion("train", self._train_confusion)
        self._train_confusion = None

    def on_validation_epoch_end(self) -> None:
        if self._val_loss_count > 0:
            self.val_epoch_loss_history.append(self._val_loss_sum / self._val_loss_count)
        self._val_loss_sum = 0.0
        self._val_loss_count = 0
        self._log_confusion("val", self._val_confusion)
        self._val_confusion = None

    def predict_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        return self(batch)

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.scheduler_step_size is None:
            return optimizer
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.scheduler_step_size,
            gamma=self.scheduler_gamma,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _shared_step(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(batch, "y"):
            raise AttributeError("Batch is missing target attribute 'y' required for training.")
        raw_preds = self._model_forward(batch)
        # If the model returns multiple outputs, use the first for loss/metrics
        preds = raw_preds[0] if isinstance(raw_preds, (tuple, list)) else raw_preds
        target = batch.y

        # Ensure target shape matches preds for BCE/CE losses
        if target.dim() == 1 and preds.dim() == 2 and target.numel() % preds.shape[-1] == 0:
            target = target.view(-1, preds.shape[-1])
        return preds, target

    @staticmethod
    def _get_batch_size(batch: Batch) -> int:
        if hasattr(batch, "num_groups"):
            return int(batch.num_groups)
        if hasattr(batch, "num_graphs"):
            return int(batch.num_graphs)
        if hasattr(batch, "batch"):
            return int(batch.batch.max().item() + 1)
        return 1

    def _compute_metrics(self, preds: torch.Tensor, target: torch.Tensor, prefix: str) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        if self.task == "classification":
            probs = torch.sigmoid(preds)
            preds_binary = (probs >= self.threshold).float()
            metrics[f"{prefix}_accuracy"] = (preds_binary == target.float()).float().mean().item()
            if preds_binary.numel() > 0:
                metrics[f"{prefix}_exact_match"] = (preds_binary == target.float()).all(dim=1).float().mean().item()
        else:
            metrics[f"{prefix}_mae"] = torch.abs(preds - target).mean().item()

        return metrics

    def _update_confusion(self, preds: torch.Tensor, target: torch.Tensor, *, is_train: bool) -> None:
        if self.task != "classification":
            return
        probs = torch.sigmoid(preds)
        preds_binary = (probs >= self.threshold).int()
        target_int = target.int()
        num_classes = target_int.shape[-1]
        if is_train:
            if self._train_confusion is None:
                self._train_confusion = torch.zeros((num_classes, 2, 2), dtype=torch.int64)
            confusion = self._train_confusion
        else:
            if self._val_confusion is None:
                self._val_confusion = torch.zeros((num_classes, 2, 2), dtype=torch.int64)
            confusion = self._val_confusion
        for cls_idx in range(num_classes):
            truth = target_int[:, cls_idx]
            pred = preds_binary[:, cls_idx]
            tn = ((truth == 0) & (pred == 0)).sum().item()
            fp = ((truth == 0) & (pred == 1)).sum().item()
            fn = ((truth == 1) & (pred == 0)).sum().item()
            tp = ((truth == 1) & (pred == 1)).sum().item()
            confusion[cls_idx, 0, 0] += int(tn)
            confusion[cls_idx, 0, 1] += int(fp)
            confusion[cls_idx, 1, 0] += int(fn)
            confusion[cls_idx, 1, 1] += int(tp)

    def _log_confusion(self, prefix: str, confusion: Optional[torch.Tensor]) -> None:
        if confusion is None:
            return
        num_classes = confusion.shape[0]
        for cls_idx in range(num_classes):
            tn, fp = confusion[cls_idx, 0].tolist()
            fn, tp = confusion[cls_idx, 1].tolist()
            total = float(tp + fp + fn)
            if total <= 0:
                continue
            self.log(f"{prefix}_tp_{cls_idx}", tp / total, on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"{prefix}_fp_{cls_idx}", fp / total, on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"{prefix}_fn_{cls_idx}", fn / total, on_step=False, on_epoch=True, prog_bar=False)

    def _model_forward(self, batch: Batch) -> torch.Tensor:
        try:
            return self.model(batch)
        except TypeError:
            pass
        if not hasattr(batch, "x") or not hasattr(batch, "edge_index") or not hasattr(batch, "edge_attr"):
            raise TypeError("Batch is missing required graph tensors for tensor-only model forward.")
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        batch_idx = getattr(batch, "batch", None)
        if batch_idx is None:
            batch_idx = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        group_ptr = getattr(batch, "group_ptr", None)
        if group_ptr is None:
            raise AttributeError("Batch is missing group_ptr for group-level classification.")
        time_group_ids = getattr(batch, "time_group_ids", None)
        if time_group_ids is None:
            raise AttributeError("Batch is missing time_group_ids for group-level classification.")
        return self.model(
            x,
            edge_index,
            edge_attr,
            batch_idx,
            group_ptr,
            time_group_ids,
        )
