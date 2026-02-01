"""Stereo-aware group classifier."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import JumpingKnowledge, AttentionalAggregation
from torch_geometric.data import Data

from pioneerml.models.base import GraphModel
from pioneerml.models.blocks import FullGraphTransformerBlock
from pioneerml.models.stereo import ViewAwareEncoder, VIEW_X_VAL, VIEW_Y_VAL


class _GroupClassifierTensorWrapper(nn.Module):
    def __init__(self, model: nn.Module):
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
        self.view_x_val = int(VIEW_X_VAL)
        self.view_y_val = int(VIEW_Y_VAL)

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

    def _forward_impl(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        group_ptr: torch.Tensor,
        time_group_ids: torch.Tensor,
        hit_mask: torch.Tensor,
    ) -> torch.Tensor:
        x_embed = self.input_embed(x)

        num_groups = int(group_ptr[-1].item())

        xs = []
        for block in self.blocks:
            x_embed = block(x_embed, edge_index, edge_attr)
            xs.append(x_embed)
        x_cat = self.jk(xs)

        raw_view = x[:, 3].long()
        mask_x = (raw_view == self.view_x_val) & hit_mask
        mask_y = (raw_view == self.view_y_val) & hit_mask

        group_id = group_ptr[batch] + time_group_ids.to(group_ptr.dtype)
        group_id = group_id.to(torch.long)

        masked_x = x_cat[mask_x]
        masked_group_id = group_id[mask_x]
        pool_x = self.pool_x(masked_x, masked_group_id, dim_size=num_groups)
        counts_x = torch.bincount(masked_group_id, minlength=num_groups).to(
            x.device,
            dtype=torch.float,
        )
        has_x = (counts_x > 0).float().unsqueeze(1)

        masked_y = x_cat[mask_y]
        masked_group_id_y = group_id[mask_y]
        pool_y = self.pool_y(masked_y, masked_group_id_y, dim_size=num_groups)
        counts_y = torch.bincount(masked_group_id_y, minlength=num_groups).to(
            x.device,
            dtype=torch.float,
        )
        has_y = (counts_y > 0).float().unsqueeze(1)

        edep = x[:, 2]
        group_edep = torch.bincount(group_id, weights=edep, minlength=num_groups).to(x.device)
        graph_u = group_edep.unsqueeze(1)
        out = torch.cat([pool_x, pool_y, graph_u, has_x, has_y], dim=1)
        return self.head(out)

    @torch.jit.ignore
    def forward(self, data: Data):
        hit_mask = getattr(data, "hit_mask", None)
        if hit_mask is None:
            hit_mask = torch.ones(data.x.size(0), dtype=torch.bool, device=data.x.device)

        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
        group_ptr = getattr(data, "group_ptr", None)
        if group_ptr is None:
            raise ValueError("group_ptr is required for group-level classification.")

        time_group_ids = getattr(data, "time_group_ids", None)
        if time_group_ids is None:
            raise ValueError("time_group_ids is required for group-level classification.")

        return self._forward_impl(
            data.x,
            data.edge_index,
            data.edge_attr,
            batch,
            group_ptr,
            time_group_ids,
            hit_mask,
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
        hit_mask = torch.ones(x.size(0), dtype=torch.bool, device=x.device)
        return self._forward_impl(
            x,
            edge_index,
            edge_attr,
            batch,
            group_ptr,
            time_group_ids,
            hit_mask=hit_mask,
        )

    @torch.jit.ignore
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

    @torch.jit.ignore
    def export_torchscript(
        self,
        path: str | "Path" | None,
        example: Data,
        *,
        strict: bool = False,
        prefer_cuda: bool = True,
    ) -> torch.jit.ScriptModule:
        if not hasattr(example, "group_ptr") or not hasattr(example, "time_group_ids"):
            raise ValueError("example Data must include group_ptr and time_group_ids.")

        wrapper = _GroupClassifierTensorWrapper(self).eval()
        device = example.x.device
        if prefer_cuda and torch.cuda.is_available():
            device = torch.device("cuda")
        wrapper = wrapper.to(device)
        example_x = example.x.to(device)
        example_edge_index = example.edge_index.to(device)
        example_edge_attr = example.edge_attr.to(device)
        example_batch = example.batch.to(device)
        example_group_ptr = example.group_ptr.to(device)
        example_time_group_ids = example.time_group_ids.to(device)
        with torch.no_grad():
            try:
                scripted = torch.jit.script(wrapper)
                if path is not None:
                    scripted.save(str(path))
                return scripted
            except Exception:
                traced = torch.jit.trace(
                    wrapper,
                    (
                        example_x,
                        example_edge_index,
                        example_edge_attr,
                        example_batch,
                        example_group_ptr,
                        example_time_group_ids,
                    ),
                    strict=strict,
                )
        if path is not None:
            traced.save(str(path))
        return traced


# Backwards-compatible alias
GroupClassifier = GroupClassifierStereo
