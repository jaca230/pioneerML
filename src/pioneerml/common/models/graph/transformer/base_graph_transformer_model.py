from __future__ import annotations

from collections.abc import Sequence

import torch.nn as nn
from torch_geometric.nn import AttentionalAggregation

from pioneerml.common.models.blocks import FullGraphTransformerBlock
from pioneerml.common.models.graph.base_graph_model import BaseGraphModel


class BaseGraphTransformerModel(BaseGraphModel):
    """Base graph model with transformer-block helper."""

    @staticmethod
    def build_transformer_blocks(*, hidden_dim: int, num_layers: int, num_heads: int, edge_dim: int, dropout: float) -> nn.ModuleList:
        return nn.ModuleList(
            [
                FullGraphTransformerBlock(
                    hidden=int(hidden_dim),
                    heads=int(num_heads),
                    edge_dim=int(edge_dim),
                    dropout=float(dropout),
                )
                for _ in range(int(num_layers))
            ]
        )

    @staticmethod
    def build_attentional_pool(*, feature_dim: int, gate_hidden_dim: int | None = None) -> AttentionalAggregation:
        hidden_dim = int(gate_hidden_dim) if gate_hidden_dim is not None else max(1, int(feature_dim) // 2)
        gate = nn.Sequential(
            nn.Linear(int(feature_dim), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        return AttentionalAggregation(gate)

    @staticmethod
    def build_mlp_head(
        *,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        dropout: float = 0.0,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        prev_dim = int(input_dim)
        hidden_dims = [int(h) for h in hidden_dims if int(h) > 0]
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, int(output_dim)))
        return nn.Sequential(*layers)
