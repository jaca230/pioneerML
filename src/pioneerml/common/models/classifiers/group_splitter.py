"""Per-hit classifier for assigning hits within a time group to particle types."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.data import Data

from pioneerml.common.models.blocks import FullGraphTransformerBlock


class GroupSplitter(nn.Module):
    """
    Per-node classifier for multi-particle hit splitting.

    Outputs per-hit logits for `[pion, muon, mip]` classes.
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
        group_probs = getattr(data, "group_probs", None)
        if group_probs is None:
            num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
            group_probs = torch.zeros((num_graphs, self.prob_dimension), device=data.x.device, dtype=data.x.dtype)
        group_total_energy = getattr(data, "group_total_energy", None)
        if group_total_energy is None:
            num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
            group_total_energy = torch.zeros((num_graphs, 1), device=data.x.device, dtype=data.x.dtype)
        return self.forward_tensors(
            data.x,
            data.edge_index,
            data.edge_attr,
            batch,
            group_total_energy,
            group_probs,
        )

    def forward_tensors(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        group_total_energy: torch.Tensor,
        group_probs: torch.Tensor,
    ):
        node_out = self.extract_embeddings_tensors(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
            group_total_energy=group_total_energy,
            group_probs=group_probs,
        )
        return self.node_head(node_out)

    def extract_embeddings_tensors(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        group_total_energy: torch.Tensor,
        group_probs: torch.Tensor,
    ) -> torch.Tensor:
        probs_expanded = group_probs[batch]
        x = self.input_proj(torch.cat([x, probs_expanded], dim=1))

        for block in self.blocks:
            x = block(x, edge_index, edge_attr)

        energy_expanded = group_total_energy[batch]
        return torch.cat([x, energy_expanded], dim=1)

    @torch.jit.ignore
    def extract_embeddings(self, data: Data) -> torch.Tensor:
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=data.x.device)
        group_probs = getattr(data, "group_probs", None)
        if group_probs is None:
            num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
            group_probs = torch.zeros((num_graphs, self.prob_dimension), device=data.x.device, dtype=data.x.dtype)
        group_total_energy = getattr(data, "group_total_energy", None)
        if group_total_energy is None:
            num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
            group_total_energy = torch.zeros((num_graphs, 1), device=data.x.device, dtype=data.x.dtype)
        return self.extract_embeddings_tensors(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            batch=batch,
            group_total_energy=group_total_energy,
            group_probs=group_probs,
        )

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
            def __init__(self, model: GroupSplitter):
                super().__init__()
                self.model = model

            def forward(
                self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                batch: torch.Tensor,
                group_total_energy: torch.Tensor,
                group_probs: torch.Tensor,
            ):
                return self.model.forward_tensors(
                    x,
                    edge_index,
                    edge_attr,
                    batch,
                    group_total_energy,
                    group_probs,
                )

        scripted = torch.jit.script(_Scriptable(self))
        if path is not None:
            scripted.save(str(path))
        return scripted
