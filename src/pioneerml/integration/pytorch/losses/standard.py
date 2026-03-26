from __future__ import annotations

import torch
import torch.nn as nn

from .base_loss import BaseLoss
from .factory.registry import REGISTRY as LOSS_REGISTRY


@LOSS_REGISTRY.register("bce_with_logits")
@LOSS_REGISTRY.register("bce")
@LOSS_REGISTRY.register("binary_cross_entropy_with_logits")
class BCEWithLogitsLoss(BaseLoss):
    @classmethod
    def from_factory(cls, *, config=None, **kwargs):
        _ = kwargs
        return cls(**dict(config or {}))

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._loss = nn.BCEWithLogitsLoss(**kwargs)

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._loss(preds, target)


@LOSS_REGISTRY.register("mse")
@LOSS_REGISTRY.register("mse_loss")
class MSELoss(BaseLoss):
    @classmethod
    def from_factory(cls, *, config=None, **kwargs):
        _ = kwargs
        return cls(**dict(config or {}))

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._loss = nn.MSELoss(**kwargs)

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._loss(preds, target)


@LOSS_REGISTRY.register("l1")
@LOSS_REGISTRY.register("mae")
class L1Loss(BaseLoss):
    @classmethod
    def from_factory(cls, *, config=None, **kwargs):
        _ = kwargs
        return cls(**dict(config or {}))

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._loss = nn.L1Loss(**kwargs)

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._loss(preds, target)
