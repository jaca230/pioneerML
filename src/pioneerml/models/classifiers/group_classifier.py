"""Stereo-aware group classifier."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import JumpingKnowledge, AttentionalAggregation
from torch_geometric.data import Data

from pioneerml.models.base import GraphModel
from pioneerml.models.blocks import FullGraphTransformerBlock
from pioneerml.models.stereo import ViewAwareEncoder, VIEW_X_VAL, VIEW_Y_VAL


class GroupClassifierStereo(GraphModel):
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

        hit_mask = getattr(data, "hit_mask", None)
        if hit_mask is None:
            hit_mask = torch.ones(x.size(0), dtype=torch.bool, device=x.device)

        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        group_ptr = getattr(data, "group_ptr", None)
        if group_ptr is None:
            raise ValueError("group_ptr is required for group-level classification.")
        num_groups = getattr(data, "num_groups", None)
        if num_groups is None:
            num_groups = int(group_ptr[-1].item())
        elif isinstance(num_groups, torch.Tensor):
            num_groups = int(num_groups.item())
        else:
            num_groups = int(num_groups)

        xs = []
        for block in self.blocks:
            x = block(x, data.edge_index, data.edge_attr)
            xs.append(x)
        x_cat = self.jk(xs)

        raw_view = data.x[:, 3].long()
        mask_x = (raw_view == VIEW_X_VAL) & hit_mask
        mask_y = (raw_view == VIEW_Y_VAL) & hit_mask

        time_group_ids = getattr(data, "time_group_ids", None)
        if time_group_ids is None:
            raise ValueError("time_group_ids is required for group-level classification.")

        group_id = group_ptr[batch] + time_group_ids.to(group_ptr.dtype)
        group_id = group_id.to(torch.long)

        def pool_and_count(mask, pool_layer):
            masked_x = x_cat[mask]
            masked_group_id = group_id[mask]
            pooled = pool_layer(masked_x, masked_group_id, dim_size=num_groups)
            counts = torch.zeros(num_groups, device=x.device)
            counts.index_add_(
                0,
                masked_group_id,
                torch.ones_like(masked_group_id, dtype=torch.float),
            )
            has_hits = (counts > 0).float().unsqueeze(1)
            return pooled, has_hits

        pool_x, has_x = pool_and_count(mask_x, self.pool_x)
        pool_y, has_y = pool_and_count(mask_y, self.pool_y)

        edep = data.x[:, 2]
        group_edep = torch.zeros(num_groups, device=x.device)
        group_edep.index_add_(0, group_id, edep)
        graph_u = group_edep.unsqueeze(1)
        out = torch.cat([pool_x, pool_y, graph_u, has_x, has_y], dim=1)
        return self.head(out)

    def extract_embeddings(self, data: Data) -> torch.Tensor:
        """
        Return the fused stereo embedding before the classification head.
        Shape: [num_groups, hidden*2 + 1 + 2]
        """
        x = self.input_embed(data.x)

        hit_mask = getattr(data, "hit_mask", None)
        if hit_mask is None:
            hit_mask = torch.ones(x.size(0), dtype=torch.bool, device=x.device)

        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        group_ptr = getattr(data, "group_ptr", None)
        if group_ptr is None:
            raise ValueError("group_ptr is required for group-level classification.")
        num_groups = getattr(data, "num_groups", None)
        if num_groups is None:
            num_groups = int(group_ptr[-1].item())
        elif isinstance(num_groups, torch.Tensor):
            num_groups = int(num_groups.item())
        else:
            num_groups = int(num_groups)

        xs = []
        for block in self.blocks:
            x = block(x, data.edge_index, data.edge_attr)
            xs.append(x)
        x_cat = self.jk(xs)

        raw_view = data.x[:, 3].long()
        mask_x = (raw_view == VIEW_X_VAL) & hit_mask
        mask_y = (raw_view == VIEW_Y_VAL) & hit_mask

        time_group_ids = getattr(data, "time_group_ids", None)
        if time_group_ids is None:
            raise ValueError("time_group_ids is required for group-level classification.")

        group_id = group_ptr[batch] + time_group_ids.to(group_ptr.dtype)
        group_id = group_id.to(torch.long)

        def pool_and_count(mask, pool_layer):
            masked_x = x_cat[mask]
            masked_group_id = group_id[mask]
            pooled = pool_layer(masked_x, masked_group_id, dim_size=num_groups)
            counts = torch.zeros(num_groups, device=x.device)
            counts.index_add_(
                0,
                masked_group_id,
                torch.ones_like(masked_group_id, dtype=torch.float),
            )
            has_hits = (counts > 0).float().unsqueeze(1)
            return pooled, has_hits

        pool_x, has_x = pool_and_count(mask_x, self.pool_x)
        pool_y, has_y = pool_and_count(mask_y, self.pool_y)

        edep = data.x[:, 2]
        group_edep = torch.zeros(num_groups, device=x.device)
        group_edep.index_add_(0, group_id, edep)
        graph_u = group_edep.unsqueeze(1)
        return torch.cat([pool_x, pool_y, graph_u, has_x, has_y], dim=1)

    def export_torchscript(
        self,
        path: str | "Path" | None,
        example: Data,
        *,
        strict: bool = False,
    ) -> torch.jit.ScriptModule:
        if not hasattr(example, "group_ptr") or not hasattr(example, "time_group_ids"):
            raise ValueError("example Data must include group_ptr and time_group_ids.")
        example_num_groups = getattr(example, "num_groups", None)
        if example_num_groups is None:
            example_num_groups = int(example.group_ptr[-1].item())
        else:
            example_num_groups = int(example_num_groups)

        class Wrapper(torch.nn.Module):
            def __init__(self, model: torch.nn.Module, num_groups: int):
                super().__init__()
                self.model = model
                self.num_groups = num_groups

            def forward(self, x, edge_index, edge_attr, batch, time_group_ids, group_ptr):
                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    time_group_ids=time_group_ids,
                )
                data.batch = batch
                data.group_ptr = group_ptr
                data.num_groups = self.num_groups
                return self.model(data)

        wrapper = Wrapper(self, example_num_groups).eval()
        with torch.no_grad():
            traced = torch.jit.trace(
                wrapper,
                (
                    example.x,
                    example.edge_index,
                    example.edge_attr,
                    example.batch,
                    example.time_group_ids,
                    example.group_ptr,
                ),
                strict=strict,
            )
        if path is not None:
            traced.save(str(path))
        return traced


# Backwards-compatible alias
GroupClassifier = GroupClassifierStereo
