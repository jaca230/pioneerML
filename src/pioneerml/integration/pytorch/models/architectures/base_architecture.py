from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class BaseArchitecture(nn.Module, ABC):
    """Generic base for all architecture plugins (graph and non-graph)."""

    @property
    def num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        if torch.jit.is_scripting():
            return 0
        return sum(p.numel() for p in self.parameters())

    @torch.jit.ignore
    def get_device(self) -> torch.device:
        """Get device of model parameters."""
        return next(self.parameters()).device

    @torch.jit.ignore
    def summary(self) -> dict[str, Any]:
        """Get architecture summary information."""
        return {
            "class": self.__class__.__name__,
            "parameters": self.num_parameters,
            "device": str(self.get_device()),
        }

    @abstractmethod
    def export_torchscript(
        self,
        path: str | Path | None,
        *,
        strict: bool = False,
    ) -> torch.jit.ScriptModule:
        """Export a TorchScript version of the architecture."""
        raise NotImplementedError

