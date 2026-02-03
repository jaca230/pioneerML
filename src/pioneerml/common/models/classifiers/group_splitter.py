"""Per-hit classifier for assigning hits within a time group to particle types."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import AttentionalAggregation

from pioneerml.common.models.blocks import FullGraphTransformerBlock


class GroupSplitter(nn.Module):
    """
    Per-node classifier for multi-particle hit splitting.

    Outputs per-hit logits for `[pion, muon, mip]` classes and regresses
    class-wise energies.
    """

    def __init__(
        self,
        in_channels: int = 4,
        prob_dimension: int = 3,
        hidden: int = 128,
        heads: int = 4,
        layers: int = 3,
        dropout: float = 0.1,
        num_classes: int = 3,
    ):
        super().__init__()
        self.layers = layers
        self.num_classes = num_classes
        self.heads = heads
        self.prob_dimension = prob_dimension

        self.input_proj = nn.Linear(in_channels + prob_dimension, hidden)

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

        self.node_head = nn.Linear(hidden + 1, num_classes)

        self.pool = AttentionalAggregation(
            nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, 1),
            )
        )

        self.energy_head = nn.Linear(hidden + 1, num_classes)

    def forward(self, data: Data):
        if hasattr(data, "group_probs") and data.group_probs is not None:
            probs_expanded = data.group_probs[data.batch]  # [N, 3]
            x = self.input_proj(torch.cat([data.x, probs_expanded], dim=1))
        else:
            if self.input_proj.in_features > data.x.shape[1]:
                padding = torch.zeros(
                    data.x.shape[0],
                    self.input_proj.in_features - data.x.shape[1],
                    device=data.x.device,
                )
                x = self.input_proj(torch.cat([data.x, padding], dim=1))
            else:
                x = self.input_proj(data.x)

        for block in self.blocks:
            x = block(x, data.edge_index, data.edge_attr)

        u_expanded = data.u[data.batch]
        node_out = torch.cat([x, u_expanded], dim=1)
        node_logits = self.node_head(node_out)

        pooled = self.pool(x, data.batch)
        graph_out = torch.cat([pooled, data.u], dim=1)
        energy_preds = self.energy_head(graph_out)

        return node_logits, energy_preds
