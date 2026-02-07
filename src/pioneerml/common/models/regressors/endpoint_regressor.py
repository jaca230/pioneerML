"""Stereo-aware endpoint regressor with quantile outputs."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import AttentionalAggregation, JumpingKnowledge

from pioneerml.common.models.blocks import FullGraphTransformerBlock
from pioneerml.common.models.components.quantile_output_head import QuantileOutputHead
from pioneerml.common.models.components.view_aware_encoder import ViewAwareEncoder


class OrthogonalEndpointRegressor(nn.Module):
    """Graph-level regressor that outputs quantile endpoints per view."""

    def __init__(
        self,
        in_channels: int = 4,
        prob_dimension: int = 3,
        hidden: int = 160,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
        quantiles=None,
    ):
        super().__init__()

        self.hit_encoder = ViewAwareEncoder(prob_dim=prob_dimension, hidden_dim=hidden)
        self.view_x_val = int(self.hit_encoder.view_x_val)
        self.view_y_val = int(self.hit_encoder.view_y_val)

        self.blocks = nn.ModuleList(
            [
                FullGraphTransformerBlock(hidden, heads=heads, edge_dim=4, dropout=dropout)
                for _ in range(layers)
            ]
        )
        self.jk = JumpingKnowledge(mode="cat")
        jk_dim = hidden * layers

        self.pool_x = AttentionalAggregation(
            nn.Sequential(nn.Linear(jk_dim, jk_dim // 2), nn.ReLU(), nn.Linear(jk_dim // 2, 1))
        )
        self.pool_y = AttentionalAggregation(
            nn.Sequential(nn.Linear(jk_dim, jk_dim // 2), nn.ReLU(), nn.Linear(jk_dim // 2, 1))
        )

        self.head_x = QuantileOutputHead(input_dim=jk_dim + 1, num_points=2, coords=1, quantiles=quantiles)
        self.head_y = QuantileOutputHead(input_dim=jk_dim + 1, num_points=2, coords=1, quantiles=quantiles)
        self.head_z = QuantileOutputHead(input_dim=jk_dim + 1, num_points=2, coords=1, quantiles=quantiles)

    def forward(self, data: Data):
        probs = data.group_probs[data.batch] if hasattr(data, "group_probs") and data.group_probs is not None else None
        x = self.hit_encoder(data.x, probs)

        xs = []
        for block in self.blocks:
            x = block(x, data.edge_index, data.edge_attr)
            xs.append(x)
        x_cat = self.jk(xs)

        raw_view = data.x[:, 3].long()
        mask_x = raw_view == self.view_x_val
        mask_y = raw_view == self.view_y_val

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

        out_x = self.head_x(torch.cat([pool_x, data.u], dim=1))
        out_y = self.head_y(torch.cat([pool_y, data.u], dim=1))

        sum_feat = (pool_x * has_x) + (pool_y * has_y)
        valid_count = (has_x + has_y).clamp(min=1.0)
        stereo_feat = sum_feat / valid_count

        out_z = self.head_z(torch.cat([stereo_feat, data.u], dim=1))

        return torch.cat([out_x, out_y, out_z], dim=2)


# Backwards-compatible alias
EndpointRegressor = OrthogonalEndpointRegressor
