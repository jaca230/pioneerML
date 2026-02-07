"""Per-hit classifier for event-level hit splitting using per-time-group priors."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.data import Data

from pioneerml.common.models.blocks import FullGraphTransformerBlock


class GroupSplitterEvent(nn.Module):
    """
    Per-node classifier for multi-particle hit splitting at event graph level.

    Uses per-time-group class priors (`group_probs`) expanded to nodes via
    `group_ptr` and `time_group_ids`.
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

    @torch.jit.ignore
    def forward(self, data: Data):
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=data.x.device)

        group_ptr = getattr(data, "group_ptr", None)
        if group_ptr is None:
            num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
            group_ptr = torch.arange(num_graphs + 1, dtype=torch.long, device=batch.device)

        time_group_ids = getattr(data, "time_group_ids", None)
        if time_group_ids is None:
            time_group_ids = torch.zeros(data.x.shape[0], dtype=torch.long, device=data.x.device)

        group_probs = getattr(data, "group_probs", None)
        if group_probs is None:
            num_groups = int(group_ptr[-1].item()) if group_ptr.numel() > 0 else 0
            group_probs = torch.zeros((num_groups, self.prob_dimension), device=data.x.device, dtype=data.x.dtype)

        return self.forward_tensors(
            data.x,
            data.edge_index,
            data.edge_attr,
            batch,
            data.u,
            group_ptr,
            time_group_ids,
            group_probs,
        )

    def forward_tensors(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        u: torch.Tensor,
        group_ptr: torch.Tensor,
        time_group_ids: torch.Tensor,
        group_probs: torch.Tensor,
    ):
        group_base = group_ptr.index_select(0, batch)
        group_ids = (group_base + time_group_ids.to(group_base.dtype)).to(torch.long)
        probs_expanded = group_probs[group_ids]

        x = self.input_proj(torch.cat([x, probs_expanded], dim=1))

        for block in self.blocks:
            x = block(x, edge_index, edge_attr)

        u_expanded = u[batch]
        node_out = torch.cat([x, u_expanded], dim=1)
        node_logits = self.node_head(node_out)

        return node_logits

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
            def __init__(self, model: GroupSplitterEvent):
                super().__init__()
                self.model = model

            def forward(
                self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                batch: torch.Tensor,
                u: torch.Tensor,
                group_ptr: torch.Tensor,
                time_group_ids: torch.Tensor,
                group_probs: torch.Tensor,
            ):
                return self.model.forward_tensors(
                    x,
                    edge_index,
                    edge_attr,
                    batch,
                    u,
                    group_ptr,
                    time_group_ids,
                    group_probs,
                )

        scripted = torch.jit.script(_Scriptable(self))
        if path is not None:
            scripted.save(str(path))
        return scripted
