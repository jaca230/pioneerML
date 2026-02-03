"""Affinity EventBuilder model."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from pioneerml.common.models.blocks import FullGraphTransformerBlock


class EventBuilder(nn.Module):
    def __init__(self, in_channels: int = 25, hidden: int = 128, heads: int = 4, layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Linear(in_channels, hidden)

        self.blocks = nn.ModuleList(
            [
                FullGraphTransformerBlock(hidden, heads=heads, edge_dim=5, dropout=dropout)
                for _ in range(layers)
            ]
        )

        self.affinity_mlp = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.output_act = nn.Sigmoid()

    def forward(self, x, edge_index, edge_attr, group_indices, batch_indices_per_group):
        """
        Args:
            x: [NumHits, 25] (Assumes col 3 is View ID: 0 for X, 1 for Y)
            group_indices: [NumHits]
            batch_indices_per_group: [NumGroups]
        """
        h = self.embedding(x)
        for block in self.blocks:
            h = block(h, edge_index, edge_attr)

        num_groups = batch_indices_per_group.size(0)

        raw_view = x[:, 3].long()
        mask_x = raw_view == 0
        mask_y = raw_view == 1

        g_max = scatter(h, group_indices, dim=0, dim_size=num_groups, reduce="max")

        def pool_view_specific(mask):
            if not mask.any():
                return (
                    torch.zeros(num_groups, h.size(1), device=h.device),
                    torch.zeros(num_groups, 1, device=h.device),
                )
            pooled = scatter(h[mask], group_indices[mask], dim=0, dim_size=num_groups, reduce="mean")
            counts = scatter(
                torch.ones_like(group_indices[mask], dtype=torch.float),
                group_indices[mask],
                dim=0,
                dim_size=num_groups,
                reduce="sum",
            )
            has_hits = (counts > 0).float().unsqueeze(1)
            return pooled, has_hits

        pool_x, has_x = pool_view_specific(mask_x)
        pool_y, has_y = pool_view_specific(mask_y)

        sum_feat = (pool_x * has_x) + (pool_y * has_y)
        valid_count = (has_x + has_y).clamp(min=1.0)
        g_stereo_mean = sum_feat / valid_count

        group_embs = torch.cat([g_stereo_mean, g_max], dim=1)

        N = num_groups
        left = group_embs.unsqueeze(1).expand(N, N, -1)
        right = group_embs.unsqueeze(0).expand(N, N, -1)

        pair_features = torch.cat([left, right], dim=-1)

        scores = self.affinity_mlp(pair_features).squeeze(-1)
        scores = (scores + scores.t()) / 2.0

        batch_ids = batch_indices_per_group.unsqueeze(1)
        event_mask = batch_ids == batch_ids.T

        probs = self.output_act(scores) * event_mask.float()

        return probs
