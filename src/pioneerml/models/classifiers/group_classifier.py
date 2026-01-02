"""Stereo-aware group classifier."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import JumpingKnowledge, AttentionalAggregation
from torch_geometric.data import Data

from pioneerml.models.blocks import FullGraphTransformerBlock
from pioneerml.models.stereo import ViewAwareEncoder, VIEW_X_VAL, VIEW_Y_VAL


class GroupClassifierStereo(nn.Module):
    def __init__(
        self,
        in_dim: int = 4,
        edge_dim: int = 4,
        hidden: int = 200,
        heads: int = 4,
        num_blocks: int = 2,
        dropout: float = 0.1,
        num_classes: int = 3,
    ):
        super().__init__()

        self.input_embed = ViewAwareEncoder(prob_dim=0, hidden_dim=hidden)

        self.blocks = nn.ModuleList(
            [
                FullGraphTransformerBlock(
                    hidden=hidden,
                    heads=heads,
                    edge_dim=edge_dim,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        self.jk = JumpingKnowledge(mode="cat")
        jk_dim = hidden * num_blocks

        self.pool_x = AttentionalAggregation(
            nn.Sequential(nn.Linear(jk_dim, jk_dim // 2), nn.ReLU(), nn.Linear(jk_dim // 2, 1))
        )
        self.pool_y = AttentionalAggregation(
            nn.Sequential(nn.Linear(jk_dim, jk_dim // 2), nn.ReLU(), nn.Linear(jk_dim // 2, 1))
        )

        concat_dim = (jk_dim * 2) + 1 + 2  # pools + global energy + valid bits

        self.head = nn.Sequential(
            nn.Linear(concat_dim, jk_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(jk_dim, jk_dim // 2),
            nn.ReLU(),
            nn.Linear(jk_dim // 2, num_classes),
        )

    def forward(self, data: Data):
        x = self.input_embed(data.x)

        xs = []
        for block in self.blocks:
            x = block(x, data.edge_index, data.edge_attr)
            xs.append(x)
        x_cat = self.jk(xs)

        raw_view = data.x[:, 3].long()
        mask_x = raw_view == VIEW_X_VAL
        mask_y = raw_view == VIEW_Y_VAL

        def pool_and_count(mask, pool_layer):
            if mask.any():
                pooled = pool_layer(x_cat[mask], data.batch[mask], dim_size=data.num_graphs)
                counts = torch.zeros(data.num_graphs, device=x.device)
                counts.index_add_(0, data.batch, mask.float())
                has_hits = (counts > 0).float().unsqueeze(1)
                return pooled, has_hits
            return (
                torch.zeros(data.num_graphs, x_cat.size(1), device=x.device),
                torch.zeros(data.num_graphs, 1, device=x.device),
            )

        pool_x, has_x = pool_and_count(mask_x, self.pool_x)
        pool_y, has_y = pool_and_count(mask_y, self.pool_y)

        out = torch.cat([pool_x, pool_y, data.u, has_x, has_y], dim=1)
        return self.head(out)


# Backwards-compatible alias
GroupClassifier = GroupClassifierStereo
