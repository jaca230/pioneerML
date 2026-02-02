"""Stereo-aware group classifier (TorchScript-friendly)."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from pioneerml.models.base import GraphModel
from pioneerml.models.blocks import FullGraphTransformerBlock
from pioneerml.models.stereo import ViewAwareEncoder, VIEW_X_VAL, VIEW_Y_VAL


class GroupClassifierStereo(GraphModel):
    def __init__(
        self,
        *,
        in_dim: int,
        edge_dim: int,
        hidden: int,
        heads: int,
        num_blocks: int,
        dropout: float,
        num_classes: int,
    ):
        super().__init__(in_channels=in_dim, hidden=hidden, edge_dim=edge_dim, dropout=dropout)

        self.input_embed = ViewAwareEncoder(prob_dim=0, hidden_dim=hidden)
        # Force these to be plain Python ints at construction time:
        self.view_x_val: int = int(VIEW_X_VAL)
        self.view_y_val: int = int(VIEW_Y_VAL)

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
        # TorchScript-friendly fallback blocks (no edge features).
        self.script_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                )
                for _ in range(num_blocks)
            ]
        )

        jk_dim = hidden * num_blocks

        concat_dim = (jk_dim * 2) + 1 + 2  # pools + global energy + valid bits

        self.head = nn.Sequential(
            nn.Linear(concat_dim, jk_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(jk_dim, jk_dim // 2),
            nn.ReLU(),
            nn.Linear(jk_dim // 2, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,           # [N, 4]
        edge_index: torch.Tensor,  # [2, E]
        edge_attr: torch.Tensor,   # [E, 4]
        batch: torch.Tensor,       # [N] in [0, num_graphs-1]
        group_ptr: torch.Tensor,   # [num_graphs+1] prefix-sum of groups per graph; group_ptr[-1] == total_groups
        time_group_ids: torch.Tensor,  # [N] in [0, groups_in_that_graph-1]
    ) -> torch.Tensor:
        x_embed = self.input_embed(x)

        # total number of groups across the whole batch:
        # NOTE: in TorchScript, this becomes an int in the scripted IR (safe).
        num_groups: int = int(group_ptr[-1].item())

        xs: List[torch.Tensor] = []
        if torch.jit.is_scripting():
            for block in self.script_blocks:
                x_embed = block(x_embed)
                xs.append(x_embed)
        else:
            for block in self.blocks:
                x_embed = block(x_embed, edge_index, edge_attr)
                xs.append(x_embed)
        x_cat = torch.cat(xs, dim=1)  # [N, jk_dim]

        raw_view = x[:, 3].to(torch.long)
        mask_x = raw_view == self.view_x_val
        mask_y = raw_view == self.view_y_val

        # group base offset per node (depends on which graph the node belongs to)
        # group_ptr[batch] is valid because batch is [N] long indices into group_ptr
        base = group_ptr.index_select(0, batch)  # [N]
        group_id = base + time_group_ids.to(base.dtype)  # [N]
        group_id = group_id.to(torch.long)

        # --- pool X (mean over nodes in X view per group) ---
        feat_dim = x_cat.size(1)
        pool_x = x_cat.new_zeros((num_groups, feat_dim))
        counts_x = x_cat.new_zeros((num_groups, 1))

        masked_x = x_cat[mask_x]
        masked_gid_x = group_id[mask_x]
        ones_x = x_cat.new_ones((masked_gid_x.size(0), 1))

        # sum
        pool_x.index_add_(0, masked_gid_x, masked_x)
        counts_x.index_add_(0, masked_gid_x, ones_x)
        # mean (avoid div-by-zero)
        pool_x = pool_x / counts_x.clamp_min(1.0)
        has_x = (counts_x > 0).to(x_cat.dtype)

        # --- pool Y (mean over nodes in Y view per group) ---
        pool_y = x_cat.new_zeros((num_groups, feat_dim))
        counts_y = x_cat.new_zeros((num_groups, 1))

        masked_y = x_cat[mask_y]
        masked_gid_y = group_id[mask_y]
        ones_y = x_cat.new_ones((masked_gid_y.size(0), 1))

        pool_y.index_add_(0, masked_gid_y, masked_y)
        counts_y.index_add_(0, masked_gid_y, ones_y)
        pool_y = pool_y / counts_y.clamp_min(1.0)
        has_y = (counts_y > 0).to(x_cat.dtype)

        # --- group energy (sum edep per group) ---
        edep = x[:, 2].to(x_cat.dtype)  # [N]
        group_edep = x_cat.new_zeros((num_groups,))
        group_edep.index_add_(0, group_id, edep)
        graph_u = group_edep.unsqueeze(1)  # [G, 1]

        out = torch.cat([pool_x, pool_y, graph_u, has_x, has_y], dim=1)
        return self.head(out)

    @torch.jit.ignore
    def export_torchscript(
        self,
        path: Optional[Union[str, "Path"]],
        *,
        prefer_cuda: bool = True,
    ) -> torch.jit.ScriptModule:
        device = self.get_device()
        if prefer_cuda and torch.cuda.is_available():
            device = torch.device("cuda")

        self.eval()
        self.to(device)
        original_blocks = self.blocks
        self.blocks = self.script_blocks
        try:
            with torch.no_grad():
                scripted = torch.jit.script(self)
        finally:
            self.blocks = original_blocks

        if path is not None:
            scripted.save(str(path))
        return scripted


GroupClassifier = GroupClassifierStereo
