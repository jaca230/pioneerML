"""Stereo-aware group classifier with event-level samples."""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch
import torch.nn as nn

from torch_geometric.nn import AttentionalAggregation

from pioneerml.common.models.base import GraphModel
from pioneerml.common.models.blocks import FullGraphTransformerBlock
from pioneerml.common.models.components.view_aware_encoder import ViewAwareEncoder


class GroupClassifierEvent(GraphModel):
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
        super().__init__(in_channels=in_dim, hidden=hidden, edge_dim=edge_dim, dropout=dropout)

        self.input_embed = ViewAwareEncoder(prob_dim=0, hidden_dim=hidden)
        self.view_x_val = int(self.input_embed.view_x_val)
        self.view_y_val = int(self.input_embed.view_y_val)

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

        jk_dim = hidden * num_blocks
        concat_dim = (jk_dim * 2) + 1 + 2

        self.pool_x_attn = AttentionalAggregation(
            nn.Sequential(
                nn.Linear(jk_dim, jk_dim // 2),
                nn.ReLU(),
                nn.Linear(jk_dim // 2, 1),
            )
        )
        self.pool_y_attn = AttentionalAggregation(
            nn.Sequential(
                nn.Linear(jk_dim, jk_dim // 2),
                nn.ReLU(),
                nn.Linear(jk_dim // 2, 1),
            )
        )

        self.head = nn.Sequential(
            nn.Linear(concat_dim, jk_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(jk_dim, jk_dim // 2),
            nn.ReLU(),
            nn.Linear(jk_dim // 2, num_classes),
        )

    @torch.jit.ignore
    def forward(self, data):
        return self.forward_tensors(
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
            data.group_ptr,
            data.time_group_ids,
        )

    def forward_tensors(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        group_ptr: torch.Tensor,
        time_group_ids: torch.Tensor,
    ) -> torch.Tensor:
        x_embed = self.input_embed(x)

        num_groups: int = int(group_ptr[-1].item())

        xs: List[torch.Tensor] = []
        for block in self.blocks:
            x_embed = block(x_embed, edge_index, edge_attr)
            xs.append(x_embed)
        x_cat = torch.cat(xs, dim=1)

        raw_view = x[:, 3].to(torch.long)
        mask_x = raw_view == self.view_x_val
        mask_y = raw_view == self.view_y_val

        base = group_ptr.index_select(0, batch)
        group_id = base + time_group_ids.to(base.dtype)
        group_id = group_id.to(torch.long)

        feat_dim = x_cat.size(1)
        masked_x = x_cat[mask_x]
        masked_gid_x = group_id[mask_x]
        masked_y = x_cat[mask_y]
        masked_gid_y = group_id[mask_y]

        counts_x = x_cat.new_zeros((num_groups, 1))
        counts_y = x_cat.new_zeros((num_groups, 1))
        ones_x = x_cat.new_ones((masked_gid_x.size(0), 1))
        ones_y = x_cat.new_ones((masked_gid_y.size(0), 1))
        counts_x.index_add_(0, masked_gid_x, ones_x)
        counts_y.index_add_(0, masked_gid_y, ones_y)
        if masked_gid_x.numel() == 0:
            pool_x = x_cat.new_zeros((num_groups, feat_dim))
        else:
            pool_x = self.pool_x_attn(masked_x, masked_gid_x, dim_size=num_groups)
        if masked_gid_y.numel() == 0:
            pool_y = x_cat.new_zeros((num_groups, feat_dim))
        else:
            pool_y = self.pool_y_attn(masked_y, masked_gid_y, dim_size=num_groups)

        has_x = (counts_x > 0).to(x_cat.dtype)
        has_y = (counts_y > 0).to(x_cat.dtype)

        edep = x[:, 2].to(x_cat.dtype)
        group_edep = x_cat.new_zeros((num_groups,))
        group_edep.index_add_(0, group_id, edep)
        graph_u = group_edep.unsqueeze(1)

        out = torch.cat([pool_x, pool_y, graph_u, has_x, has_y], dim=1)
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
            def __init__(self, model: GroupClassifierEvent):
                super().__init__()
                self.model = model

            def forward(
                self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                batch: torch.Tensor,
                group_ptr: torch.Tensor,
                time_group_ids: torch.Tensor,
            ) -> torch.Tensor:
                return self.model.forward_tensors(
                    x,
                    edge_index,
                    edge_attr,
                    batch,
                    group_ptr,
                    time_group_ids,
                )

        scriptable = _Scriptable(self)
        scripted = torch.jit.script(scriptable)

        if path is not None:
            scripted.save(str(path))
        return scripted


GroupClassifierEventStereo = GroupClassifierEvent
