"""Time-group endpoint regressor for per-group start/end coordinates."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.data import Data

from pioneerml.common.models.blocks import FullGraphTransformerBlock


class EndpointRegressor(nn.Module):
    """Predicts `[start_x, start_y, start_z, end_x, end_y, end_z]` per time-group graph."""

    def __init__(
        self,
        in_channels: int = 4,
        group_prob_dimension: int = 3,
        splitter_prob_dimension: int = 3,
        hidden: int = 192,
        heads: int = 4,
        layers: int = 3,
        dropout: float = 0.1,
        output_dim: int = 6,
    ):
        super().__init__()
        self.group_prob_dimension = int(group_prob_dimension)
        self.splitter_prob_dimension = int(splitter_prob_dimension)

        input_dim = int(in_channels) + self.group_prob_dimension + self.splitter_prob_dimension
        self.input_proj = nn.Linear(input_dim, hidden)

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

        pooled_dim = hidden * 3 + self.group_prob_dimension + 3
        self.head = nn.Sequential(
            nn.Linear(pooled_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, output_dim),
        )

    @torch.jit.ignore
    def forward(self, data: Data) -> torch.Tensor:
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=data.x.device)

        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0

        u = getattr(data, "u", None)
        if u is None:
            u = torch.zeros((num_graphs, 1), dtype=data.x.dtype, device=data.x.device)

        group_probs = getattr(data, "group_probs", None)
        if group_probs is None:
            group_probs = torch.zeros(
                (num_graphs, self.group_prob_dimension),
                dtype=data.x.dtype,
                device=data.x.device,
            )

        splitter_probs = getattr(data, "splitter_probs", None)
        if splitter_probs is None:
            splitter_probs = torch.zeros(
                (data.x.shape[0], self.splitter_prob_dimension),
                dtype=data.x.dtype,
                device=data.x.device,
            )

        return self.forward_tensors(
            data.x,
            data.edge_index,
            data.edge_attr,
            batch,
            u,
            group_probs,
            splitter_probs,
        )

    def forward_tensors(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        u: torch.Tensor,
        group_probs: torch.Tensor,
        splitter_probs: torch.Tensor,
    ) -> torch.Tensor:
        raw_x = x
        probs_expanded = group_probs[batch]

        if splitter_probs.numel() == 0:
            splitter_probs = torch.zeros(
                (x.size(0), self.splitter_prob_dimension),
                device=x.device,
                dtype=x.dtype,
            )

        x = self.input_proj(torch.cat([raw_x, probs_expanded, splitter_probs], dim=1))

        for block in self.blocks:
            x = block(x, edge_index, edge_attr)

        num_graphs = int(u.shape[0])
        feat_dim = int(x.size(1))

        pool_x = x.new_zeros((num_graphs, feat_dim))
        pool_y = x.new_zeros((num_graphs, feat_dim))
        pool_all = x.new_zeros((num_graphs, feat_dim))
        counts_x = x.new_zeros((num_graphs, 1))
        counts_y = x.new_zeros((num_graphs, 1))
        counts_all = x.new_zeros((num_graphs, 1))

        one_vec = x.new_ones((x.size(0), 1))
        counts_all.index_add_(0, batch, one_vec)
        pool_all.index_add_(0, batch, x)

        raw_view = raw_x[:, 3].to(torch.long)
        mask_x = raw_view == 0
        mask_y = raw_view == 1

        if bool(mask_x.any().item()):
            bid_x = batch[mask_x]
            x_x = x[mask_x]
            pool_x.index_add_(0, bid_x, x_x)
            counts_x.index_add_(0, bid_x, x_x.new_ones((x_x.size(0), 1)))
        if bool(mask_y.any().item()):
            bid_y = batch[mask_y]
            x_y = x[mask_y]
            pool_y.index_add_(0, bid_y, x_y)
            counts_y.index_add_(0, bid_y, x_y.new_ones((x_y.size(0), 1)))

        pool_all = pool_all / counts_all.clamp_min(1.0)
        pool_x = pool_x / counts_x.clamp_min(1.0)
        pool_y = pool_y / counts_y.clamp_min(1.0)

        has_x = (counts_x > 0).to(x.dtype)
        has_y = (counts_y > 0).to(x.dtype)

        out = torch.cat([pool_x, pool_y, pool_all, group_probs, u.to(x.dtype), has_x, has_y], dim=1)
        return self.head(out)

    def export_torchscript(
        self,
        path: str | Path | None,
        *,
        prefer_cuda: bool = True,
        strict: bool = False,
    ) -> torch.jit.ScriptModule:
        device = torch.device("cuda") if prefer_cuda and torch.cuda.is_available() else torch.device("cpu")
        self.eval()
        self.to(device)
        _ = strict

        class _Scriptable(nn.Module):
            def __init__(self, model: EndpointRegressor):
                super().__init__()
                self.model = model

            def forward(
                self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                batch: torch.Tensor,
                u: torch.Tensor,
                group_probs: torch.Tensor,
                splitter_probs: torch.Tensor,
            ) -> torch.Tensor:
                return self.model.forward_tensors(
                    x,
                    edge_index,
                    edge_attr,
                    batch,
                    u,
                    group_probs,
                    splitter_probs,
                )

        scripted = torch.jit.script(_Scriptable(self))
        if path is not None:
            scripted.save(str(path))
        return scripted


OrthogonalEndpointRegressor = EndpointRegressor
