"""
Regressor for predicting pion stop position from a time-group graph.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import AttentionalAggregation, JumpingKnowledge

from pioneerml.models.base import GraphModel
from pioneerml.models.blocks import FullGraphTransformerBlock


class PionStopRegressor(GraphModel):
    """Graph-level regressor for the pion stop 3D coordinate."""

    def __init__(
        self,
        in_channels: int = 5,
        hidden: int = 128,
        heads: int = 4,
        layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__(
            in_channels=in_channels,
            hidden=hidden,
            edge_dim=4,
            dropout=dropout,
        )
        self.layers = layers
        self.heads = heads

        self.input_proj = nn.Linear(in_channels, hidden)

        self.blocks = nn.ModuleList(
            [
                FullGraphTransformerBlock(
                    hidden=hidden,
                    heads=heads,
                    edge_dim=4,
                    dropout=dropout,
                )
                for _ in range(layers)
            ]
        )

        self.jk = JumpingKnowledge(mode="cat")
        jk_dim = hidden * layers

        self.pool = AttentionalAggregation(
            nn.Sequential(
                nn.Linear(jk_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )
        )

        self.head = nn.Sequential(
            nn.Linear(jk_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass returning a 3D coordinate for each graph."""
        x = self.input_proj(data.x)
        xs = []
        for block in self.blocks:
            x = block(x, data.edge_index, data.edge_attr)
            xs.append(x)
        x_cat = self.jk(xs)
        pooled = self.pool(x_cat, data.batch)
        return self.head(pooled)

    def summary(self) -> dict:
        info = super().summary()
        info.update(
            {
                "layers": self.layers,
                "heads": self.heads,
            }
        )
        return info
