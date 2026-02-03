"""Regressor for predicting positron angle components from a time-group graph."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import AttentionalAggregation, JumpingKnowledge

from pioneerml.common.models.blocks import FullGraphTransformerBlock


class PositronAngleModel(nn.Module):
    """Graph-level regressor outputting 3-vector angle prediction per graph."""

    def __init__(
        self,
        in_channels: int = 4,
        hidden: int = 128,
        heads: int = 4,
        layers: int = 2,
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
            nn.Linear(jk_dim + 3, hidden),  # Adds pion stop position
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

        # Prefer pred_pion_stop, fall back to pion_stop attribute if provided
        if hasattr(data, "pred_pion_stop"):
            pion_stop = data.pred_pion_stop
        elif hasattr(data, "pion_stop"):
            pion_stop = data.pion_stop
        else:
            raise AttributeError("PositronAngleModel expects pred_pion_stop or pion_stop on the Data object.")

        return self.head(torch.cat([pooled, pion_stop], dim=1))
