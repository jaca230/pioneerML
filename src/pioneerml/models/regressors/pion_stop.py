"""Regressor for predicting pion stop position from a time-group graph."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import AttentionalAggregation, JumpingKnowledge

from pioneerml.models.blocks import FullGraphTransformerBlock


class PionStopRegressor(nn.Module):
    """Graph-level regressor for the pion stop 3D coordinate."""

    def __init__(
        self,
        in_channels: int = 4,
        hidden: int = 128,
        heads: int = 4,
        layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
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
            nn.Linear(jk_dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x = self.input_proj(data.x)
        xs = []
        for block in self.blocks:
            x = block(x, data.edge_index, data.edge_attr)
            xs.append(x)
        x_cat = self.jk(xs)
        pooled = self.pool(x_cat, data.batch)
        out = torch.cat([pooled, data.u], dim=1)
        return self.head(out)

    def extract_embeddings(self, data: Data) -> torch.Tensor:
        x = self.input_proj(data.x)
        xs = []
        for block in self.blocks:
            x = block(x, data.edge_index, data.edge_attr)
            xs.append(x)
        x_cat = self.jk(xs)
        pooled = self.pool(x_cat, data.batch)
        return torch.cat([pooled, data.u], dim=1)
