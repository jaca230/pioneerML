"""Affinity scorer for evaluating group consistency."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import AttentionalAggregation, JumpingKnowledge

from pioneerml.models.blocks import FullGraphTransformerBlock


class GroupAffinityModel(nn.Module):
    """
    Graph-level affinity scorer.
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: int = 128,
        heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.heads = heads

        self.input_proj = nn.Linear(in_channels, hidden_channels)

        self.layers = nn.ModuleList(
            [
                FullGraphTransformerBlock(
                    hidden=hidden_channels,
                    heads=heads,
                    edge_dim=4,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.jk = JumpingKnowledge(mode="cat")
        jk_dim = hidden_channels * num_layers

        self.pool = AttentionalAggregation(
            nn.Sequential(
                nn.Linear(jk_dim, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, 1),
            )
        )

        self.head = nn.Sequential(
            nn.Linear(jk_dim + 1, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x = self.input_proj(data.x)
        xs = []
        for block in self.layers:
            x = block(x, data.edge_index, data.edge_attr)
            xs.append(x)
        x_cat = self.jk(xs)
        pooled = self.pool(x_cat, data.batch)
        out = torch.cat([pooled, data.u], dim=1)
        return self.head(out)
