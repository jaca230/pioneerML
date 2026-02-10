"""Event-level edge-affinity splitter model."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.data import Data

from pioneerml.common.models.blocks import FullGraphTransformerBlock


class EventSplitter(nn.Module):
    """Predicts per-edge affinity logits for event-level hit graphs."""

    def __init__(
        self,
        in_channels: int = 4,
        group_prob_dimension: int = 3,
        splitter_prob_dimension: int = 3,
        endpoint_dimension: int = 6,
        edge_attr_dimension: int = 4,
        hidden: int = 192,
        heads: int = 4,
        layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.group_prob_dimension = int(group_prob_dimension)
        self.splitter_prob_dimension = int(splitter_prob_dimension)
        self.endpoint_dimension = int(endpoint_dimension)
        self.edge_attr_dimension = int(edge_attr_dimension)

        input_dim = (
            int(in_channels)
            + self.group_prob_dimension
            + self.splitter_prob_dimension
            + self.endpoint_dimension
        )
        self.input_proj = nn.Linear(input_dim, hidden)

        self.blocks = nn.ModuleList(
            [
                FullGraphTransformerBlock(
                    hidden=hidden,
                    heads=heads,
                    edge_dim=self.edge_attr_dimension,
                    dropout=dropout,
                )
                for _ in range(layers)
            ]
        )

        self.edge_head = nn.Sequential(
            nn.Linear((hidden * 2) + self.edge_attr_dimension, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    @torch.jit.ignore
    def forward(self, data: Data) -> torch.Tensor:
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
            group_probs = torch.zeros(
                (num_groups, self.group_prob_dimension),
                device=data.x.device,
                dtype=data.x.dtype,
            )

        splitter_probs = getattr(data, "splitter_probs", None)
        if splitter_probs is None:
            splitter_probs = torch.zeros(
                (data.x.shape[0], self.splitter_prob_dimension),
                device=data.x.device,
                dtype=data.x.dtype,
            )

        endpoint_preds = getattr(data, "endpoint_preds", None)
        if endpoint_preds is None:
            num_groups = int(group_ptr[-1].item()) if group_ptr.numel() > 0 else 0
            endpoint_preds = torch.zeros(
                (num_groups, self.endpoint_dimension),
                device=data.x.device,
                dtype=data.x.dtype,
            )

        return self.forward_tensors(
            data.x,
            data.edge_index,
            data.edge_attr,
            batch,
            group_ptr,
            time_group_ids,
            group_probs,
            splitter_probs,
            endpoint_preds,
        )

    def forward_tensors(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        group_ptr: torch.Tensor,
        time_group_ids: torch.Tensor,
        group_probs: torch.Tensor,
        splitter_probs: torch.Tensor,
        endpoint_preds: torch.Tensor,
    ) -> torch.Tensor:
        group_base = group_ptr.index_select(0, batch) if group_ptr.numel() > 0 else batch.new_zeros(batch.shape)
        group_ids = (group_base + time_group_ids.to(group_base.dtype)).to(torch.long)

        if group_probs.numel() == 0:
            expanded_group_probs = torch.zeros(
                (x.size(0), self.group_prob_dimension),
                device=x.device,
                dtype=x.dtype,
            )
        else:
            max_group = int(group_probs.size(0) - 1)
            group_ids = group_ids.clamp(min=0, max=max_group)
            expanded_group_probs = group_probs[group_ids]

        if splitter_probs.numel() == 0:
            splitter_probs = torch.zeros(
                (x.size(0), self.splitter_prob_dimension),
                device=x.device,
                dtype=x.dtype,
            )

        if endpoint_preds.numel() == 0:
            expanded_endpoint_preds = torch.zeros(
                (x.size(0), self.endpoint_dimension),
                device=x.device,
                dtype=x.dtype,
            )
        else:
            max_group = int(endpoint_preds.size(0) - 1)
            safe_group_ids = group_ids.clamp(min=0, max=max_group)
            expanded_endpoint_preds = endpoint_preds[safe_group_ids]

        h = self.input_proj(
            torch.cat(
                [x, expanded_group_probs, splitter_probs, expanded_endpoint_preds],
                dim=1,
            )
        )

        for block in self.blocks:
            h = block(h, edge_index, edge_attr)

        if edge_index.numel() == 0:
            return h.new_zeros((0, 1))

        src = edge_index[0].to(torch.long)
        dst = edge_index[1].to(torch.long)
        pair_features = torch.cat([h[src], h[dst], edge_attr], dim=1)
        return self.edge_head(pair_features)

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
            def __init__(self, model: EventSplitter):
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
                group_probs: torch.Tensor,
                splitter_probs: torch.Tensor,
                endpoint_preds: torch.Tensor,
            ) -> torch.Tensor:
                return self.model.forward_tensors(
                    x,
                    edge_index,
                    edge_attr,
                    batch,
                    group_ptr,
                    time_group_ids,
                    group_probs,
                    splitter_probs,
                    endpoint_preds,
                )

        scripted = torch.jit.script(_Scriptable(self))
        if path is not None:
            scripted.save(str(path))
        return scripted
