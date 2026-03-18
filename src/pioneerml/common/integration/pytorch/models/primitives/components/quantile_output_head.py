"""Quantile projection head used by endpoint regressors."""

from __future__ import annotations

import torch
import torch.nn as nn


class QuantileOutputHead(nn.Module):
    def __init__(self, input_dim, num_points=2, coords=3, quantiles=None):
        super().__init__()
        if quantiles is None:
            quantiles = [0.16, 0.50, 0.84]
        self.quantiles = sorted(quantiles)
        self.mid_index = self.quantiles.index(0.50)
        self.num_points = num_points
        self.coords = coords
        self.num_quantiles = len(quantiles)
        self.projection = nn.Linear(input_dim, num_points * coords * self.num_quantiles)

    def forward(self, x):
        batch_size = x.shape[0]
        raw = self.projection(x)
        raw = raw.view(batch_size, self.num_points, self.coords, self.num_quantiles)

        median = raw[..., self.mid_index]

        upper_offsets = torch.nn.functional.softplus(raw[..., self.mid_index + 1 :])
        upper_vals = median.unsqueeze(-1) + torch.cumsum(upper_offsets, dim=-1)

        lower_offsets = torch.nn.functional.softplus(raw[..., : self.mid_index])
        lower_offsets_flipped = torch.flip(lower_offsets, dims=[-1])
        lower_vals = median.unsqueeze(-1) - torch.cumsum(lower_offsets_flipped, dim=-1)
        lower_vals = torch.flip(lower_vals, dims=[-1])

        return torch.cat([lower_vals, median.unsqueeze(-1), upper_vals], dim=-1)


__all__ = ["QuantileOutputHead"]
