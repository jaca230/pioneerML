"""
Per-hit classifier for assigning hits within a time group to particle types.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data

from pioneerml.models.base import GraphModel
from pioneerml.models.blocks import FullGraphTransformerBlock


class GroupSplitter(GraphModel):
    """
    Per-node classifier for multi-particle hit splitting.

    Outputs per-hit logits for `[pion, muon, mip]` classes.
    """

    def __init__(
        self,
        in_channels: int = 5,
        hidden: int = 128,
        heads: int = 4,
        layers: int = 3,
        dropout: float = 0.1,
        num_classes: int = 3,
    ):
        super().__init__(
            in_channels=in_channels,
            hidden=hidden,
            edge_dim=4,
            dropout=dropout,
        )
        self.layers = layers
        self.num_classes = num_classes
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

        self.head = nn.Linear(hidden, num_classes)

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass returning per-node logits."""
        x = self.input_proj(data.x)
        for block in self.blocks:
            x = block(x, data.edge_index, data.edge_attr)
        return self.head(x)

    def summary(self) -> dict:
        info = super().summary()
        info.update(
            {
                "layers": self.layers,
                "num_classes": self.num_classes,
                "heads": self.heads,
            }
        )
        return info
