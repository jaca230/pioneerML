from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GraphTensorDims:
    """Unified dimension config for graph loaders."""

    node_feature_dim: int
    edge_feature_dim: int
    graph_feature_dim: int = 0
    node_target_dim: int = 0
    edge_target_dim: int = 0
    graph_target_dim: int = 0

