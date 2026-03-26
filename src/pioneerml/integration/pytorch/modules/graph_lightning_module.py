"""
Lightweight PyTorch Lightning utilities for graph models.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Batch

from pioneerml.integration.pytorch.losses import LossFactory
from pioneerml.integration.pytorch.models.architectures.graph import BaseGraphModel
from .factory.registry import REGISTRY as MODULE_REGISTRY


@MODULE_REGISTRY.register("graph_lightning")
class GraphLightningModule(pl.LightningModule):
    """
    Generic LightningModule wrapper for BaseGraphModel instances.

    Handles common training/validation loops while delegating the forward pass
    to the underlying graph model.

    Loss contract:
      - Preferred: ``loss_fn(predictions, batch) -> loss`` or
        ``loss_fn(predictions, batch) -> (loss, terms_dict)``
      - Backward compatible: ``loss_fn(preds_tensor, target_tensor) -> loss``
    """

    def __init__(
        self,
        model: BaseGraphModel,
        *,
        loss_fn: Callable[..., Any],
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer_cls: type[optim.Optimizer] = optim.AdamW,
        scheduler_step_size: int | None = None,
        scheduler_gamma: float = 0.5,
        max_step_history: int | None = 2048,
        max_epoch_history: int | None = 512,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_cls = optimizer_cls
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.max_step_history = self._resolve_history_limit(max_step_history)
        self.max_epoch_history = self._resolve_history_limit(max_epoch_history)

        # Loss histories (bounded by default to avoid unbounded RAM growth).
        self.train_loss_history: list[float] = []
        self.val_loss_history: list[float] = []
        self.train_epoch_loss_history: list[float] = []
        self.val_epoch_loss_history: list[float] = []
        self._train_loss_sum: float = 0.0
        self._train_loss_count: int = 0
        self._val_loss_sum: float = 0.0
        self._val_loss_count: int = 0

    @classmethod
    def from_factory(cls, *, config: Mapping[str, Any] | None = None, **kwargs):
        merged = {**dict(config or {}), **dict(kwargs)}
        merged.pop("namespace", None)
        merged.pop("name", None)
        merged.pop("config", None)

        if "model" not in merged:
            raise ValueError("graph_lightning module config must include 'model'.")

        loss_spec = merged.get("loss")
        if loss_spec is None:
            loss_spec = merged.get("loss_fn")
        if loss_spec is None:
            raise ValueError("graph_lightning module config must include 'loss' or 'loss_fn'.")
        merged["loss_fn"] = cls._resolve_loss_fn(loss_spec)
        merged.pop("loss", None)
        return cls(**merged)

    @staticmethod
    def _resolve_loss_fn(spec: Any) -> Callable[..., Any]:
        if callable(spec):
            return spec

        if isinstance(spec, str):
            return LossFactory(loss_name=spec).build(config={})

        if isinstance(spec, Mapping):
            block = dict(spec)
            if "type" in block:
                loss_type = block.get("type")
                if not isinstance(loss_type, str) or loss_type.strip() == "":
                    raise ValueError("module.loss.type must be a non-empty string.")
                loss_cfg = block.get("config", {})
                if not isinstance(loss_cfg, Mapping):
                    raise TypeError("module.loss.config must be a mapping when provided.")
                return LossFactory(loss_name=loss_type).build(config=dict(loss_cfg))
            value = block.get("value")
            if callable(value):
                return value
            raise TypeError("module.loss fixed loss requires callable 'value'.")

        raise TypeError("module.loss must be callable, string, or mapping with keys ['type', 'config'].")

    def forward(self, batch: Batch) -> torch.Tensor:
        return self._model_forward(batch)

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        raw_preds = self._model_forward(batch)
        loss, terms = self.compute_loss(raw_preds, batch)
        bs = self._get_batch_size(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        for key, value in terms.items():
            if key == "loss":
                continue
            self.log(f"train_{key}", value, on_step=False, on_epoch=True, prog_bar=False, batch_size=bs)
        self._append_history(
            self.train_loss_history,
            float(loss.detach().cpu().item()),
            max_points=self.max_step_history,
        )
        self._train_loss_sum += loss.detach().cpu().item() * bs
        self._train_loss_count += bs
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        raw_preds = self._model_forward(batch)
        loss, terms = self.compute_loss(raw_preds, batch)
        bs = self._get_batch_size(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        for key, value in terms.items():
            if key == "loss":
                continue
            self.log(f"val_{key}", value, on_step=False, on_epoch=True, prog_bar=False, batch_size=bs)
        self._append_history(
            self.val_loss_history,
            float(loss.detach().cpu().item()),
            max_points=self.max_step_history,
        )
        self._val_loss_sum += loss.detach().cpu().item() * bs
        self._val_loss_count += bs

    def on_train_epoch_end(self) -> None:
        if self._train_loss_count > 0:
            self._append_history(
                self.train_epoch_loss_history,
                float(self._train_loss_sum / self._train_loss_count),
                max_points=self.max_epoch_history,
            )
        self._train_loss_sum = 0.0
        self._train_loss_count = 0

    def on_validation_epoch_end(self) -> None:
        if self._val_loss_count > 0:
            self._append_history(
                self.val_epoch_loss_history,
                float(self._val_loss_sum / self._val_loss_count),
                max_points=self.max_epoch_history,
            )
        self._val_loss_sum = 0.0
        self._val_loss_count = 0

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

    @staticmethod
    def primary_predictions(raw_preds: Any) -> torch.Tensor:
        preds = raw_preds[0] if isinstance(raw_preds, (tuple, list)) else raw_preds
        if not isinstance(preds, torch.Tensor):
            raise TypeError("Primary predictions must be a torch.Tensor for this operation.")
        return preds

    @staticmethod
    def primary_target(batch: Batch, preds: torch.Tensor) -> torch.Tensor:
        candidate_fields = ("y_graph", "y_node", "y_edge", "y")
        target = None
        for field in candidate_fields:
            value = getattr(batch, field, None)
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                if value.dim() >= 1 and preds.dim() >= 1 and int(value.shape[0]) == int(preds.shape[0]):
                    target = value
                    break
                if field == "y" and target is None:
                    target = value
        if target is None:
            raise AttributeError("Batch is missing usable target tensor (expected one of y_graph/y_node/y_edge/y).")

        # Ensure target shape matches preds for BCE/CE losses
        if target.dim() == 1 and preds.dim() == 2 and target.numel() % preds.shape[-1] == 0:
            target = target.view(-1, preds.shape[-1])
        return target

    @staticmethod
    def _normalize_loss_output(loss_output: Any) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if isinstance(loss_output, tuple) and len(loss_output) == 2 and isinstance(loss_output[1], dict):
            loss = loss_output[0]
            terms = dict(loss_output[1])
        else:
            loss = loss_output
            terms = {}
        if not isinstance(loss, torch.Tensor):
            raise TypeError("loss_fn must return a torch.Tensor loss.")
        if "loss" not in terms:
            terms["loss"] = loss
        return loss, terms

    def compute_loss(self, raw_preds: Any, batch: Batch) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Preferred contract: loss_fn(predictions, batch)
        if not isinstance(self.loss_fn, torch.nn.Module):
            try:
                out = self.loss_fn(raw_preds, batch)
                return self._normalize_loss_output(out)
            except TypeError:
                pass

        # Backward-compatible contract: loss_fn(preds_tensor, target_tensor)
        preds = self.primary_predictions(raw_preds)
        target = self.primary_target(batch, preds)
        out = self.loss_fn(preds, target)
        return self._normalize_loss_output(out)

    @staticmethod
    def _get_batch_size(batch: Batch) -> int:
        if hasattr(batch, "num_groups"):
            return int(batch.num_groups)
        if hasattr(batch, "num_graphs"):
            return int(batch.num_graphs)
        if hasattr(batch, "node_graph_id"):
            node_graph_id = getattr(batch, "node_graph_id")
            if node_graph_id is not None and int(node_graph_id.numel()) > 0:
                return int(node_graph_id.max().item() + 1)
        if hasattr(batch, "batch"):
            return int(batch.batch.max().item() + 1)
        return 1

    @staticmethod
    def _resolve_history_limit(value: int | None) -> int | None:
        if value is None:
            return None
        out = int(value)
        if out < 0:
            raise ValueError("History limits must be >= 0 when provided.")
        return out

    @staticmethod
    def _append_history(history: list[float], value: float, *, max_points: int | None) -> None:
        if max_points == 0:
            return
        history.append(float(value))
        if max_points is not None and len(history) > max_points:
            del history[: len(history) - max_points]

    def _model_forward(self, batch: Batch) -> torch.Tensor:
        return self.model(batch)
