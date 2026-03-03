"""
Base class for graph neural network models.

All PIONEER ML models inherit from BaseGraphModel to ensure
a consistent interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
from torch_geometric.data import Data


class BaseGraphModel(nn.Module, ABC):
    """
    Abstract base class for graph neural network models.

    All PIONEER ML models should inherit from this class to ensure
    consistent interfaces and common functionality.

    Standard graph batch contract:
    - Inputs: ``x_node``, ``x_edge``, ``x_graph``
    - Targets: ``y_node``, ``y_edge``, ``y_graph``
    - Topology: ``edge_index``
    - Ownership: ``node_graph_id`` (and optional ``edge_graph_id``)
    - Layout pointers: ``graph_ptr``, ``node_ptr``, ``edge_ptr``

    Subclasses must implement the forward() method.

    Attributes:
        node_dim: Number of input node feature channels (default: 5).
        hidden: Hidden dimension size.
        edge_dim: Number of edge feature channels (default: 4).
        graph_dim: Number of graph-level feature channels (default: 0).
        dropout: Dropout rate.

    Example:
        >>> class MyModel(BaseGraphModel):
        ...     def forward(self, data):
        ...         # Implement forward pass
        ...         return predictions
    """

    def __init__(
        self,
        node_dim: int = 5,
        edge_dim: int = 4,
        graph_dim: int = 0,
        hidden: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.graph_dim = int(graph_dim)
        self.hidden = int(hidden)
        self.dropout = float(dropout)

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
            "node_dim": self.node_dim,
            "hidden": self.hidden,
            "edge_dim": self.edge_dim,
            "graph_dim": self.graph_dim,
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

    @staticmethod
    def node_features(data: Data) -> torch.Tensor:
        x = getattr(data, "x_node", None)
        if x is None:
            raise AttributeError("Batch is missing node features ('x_node').")
        return x

    @staticmethod
    def edge_features(data: Data) -> torch.Tensor:
        edge_attr = getattr(data, "x_edge", None)
        if edge_attr is None:
            raise AttributeError("Batch is missing edge features ('x_edge').")
        return edge_attr

    @staticmethod
    def graph_features(data: Data) -> torch.Tensor:
        graph_features = getattr(data, "x_graph", None)
        if graph_features is None:
            raise AttributeError("Batch is missing graph features ('x_graph').")
        return graph_features

    @staticmethod
    def node_graph_id(data: Data) -> torch.Tensor:
        batch = getattr(data, "node_graph_id", None)
        if batch is None:
            raise AttributeError("Batch is missing node-graph ownership ('node_graph_id').")
        return batch


GraphModel = BaseGraphModel
