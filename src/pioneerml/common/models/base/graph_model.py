"""
Base class for graph neural network models.

All PIONEER ML models inherit from GraphModel to ensure
a consistent interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
from torch_geometric.data import Data


class GraphModel(nn.Module, ABC):
    """
    Abstract base class for graph neural network models.

    All PIONEER ML models should inherit from this class to ensure
    consistent interfaces and common functionality.

    Standard graph features:
    - Node features (5D): [coord, z, energy, view, group_energy]
    - Edge features (4D): [dx, dz, dE, same_view]

    Subclasses must implement the forward() method.

    Attributes:
        in_channels: Number of input node feature channels (default: 5).
        hidden: Hidden dimension size.
        edge_dim: Number of edge feature channels (default: 4).
        dropout: Dropout rate.

    Example:
        >>> class MyModel(GraphModel):
        ...     def forward(self, data):
        ...         # Implement forward pass
        ...         return predictions
    """

    def __init__(
        self,
        in_channels: int = 5,
        hidden: int = 128,
        edge_dim: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden = hidden
        self.edge_dim = edge_dim
        self.dropout = dropout

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
    def summary(self) -> dict:
        """
        Get model summary information.

        Returns:
            Dictionary with model info.
        """
        return {
            "class": self.__class__.__name__,
            "parameters": self.num_parameters,
            "device": str(self.get_device()),
            "in_channels": self.in_channels,
            "hidden": self.hidden,
            "edge_dim": self.edge_dim,
            "dropout": self.dropout,
        }

    @abstractmethod
    def export_torchscript(
        self,
        path: str | Path | None,
        *,
        strict: bool = False,
    ) -> torch.jit.ScriptModule:
        """Export a TorchScript version of the model."""
        raise NotImplementedError
