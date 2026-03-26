from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseLoss(nn.Module, ABC):
    """Base class for loss plugins."""

    @abstractmethod
    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

