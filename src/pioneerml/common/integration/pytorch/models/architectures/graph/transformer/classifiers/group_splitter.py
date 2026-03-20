"""Per-hit classifier for assigning hits within a time group to particle types."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.data import Data

from pioneerml.common.integration.pytorch.models.architectures.graph.transformer.classifiers.base_graph_classifier_model import BaseGraphClassifierModel
from pioneerml.common.integration.pytorch.models.architectures.factory.registry import REGISTRY as ARCHITECTURE_REGISTRY


@ARCHITECTURE_REGISTRY.register("group_splitter")
class GroupSplitter(BaseGraphClassifierModel):
    """
    Per-node classifier for multi-particle hit splitting.

    Outputs per-hit logits for `[pion, muon, mip]` classes.
    """

    def __init__(
        self,
        node_dim: int = 4,
        edge_dim: int = 4,
        graph_dim: int = 3,
        hidden: int = 128,
        heads: int = 4,
        layers: int = 3,
        dropout: float = 0.1,
        num_classes: int = 3,
    ):
        super().__init__(node_dim=node_dim, edge_dim=edge_dim, graph_dim=graph_dim, hidden=hidden, dropout=dropout)
        self.layers = layers
        self.num_classes = num_classes
        self.heads = heads
        self.prob_dimension = int(graph_dim)

        self.input_proj = nn.Linear(int(node_dim) + int(self.prob_dimension), hidden)

        self.blocks = self.build_transformer_blocks(
            hidden_dim=hidden,
            num_layers=layers,
            num_heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
        )

        self.node_head = nn.Linear(hidden, num_classes)

    def _graph_features(self, data: Data, *, batch: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        group_probs = getattr(data, "group_probs", None)
        if group_probs is None:
            try:
                x_graph = self.graph_features(data)
            except AttributeError:
                x_graph = None
            if x_graph is not None and x_graph.dim() == 2 and int(x_graph.shape[1]) >= int(self.prob_dimension):
                group_probs = x_graph[:, : self.prob_dimension]
        if group_probs is None:
            num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
            group_probs = torch.zeros((num_graphs, self.prob_dimension), device=device, dtype=dtype)
        return group_probs

    @torch.jit.ignore
    def forward(self, data: Data):
        x = self.node_features(data)
        edge_attr = self.edge_features(data)
        batch = self.node_graph_id(data)
        group_probs = self._graph_features(data, batch=batch, dtype=x.dtype, device=x.device)
        return self.forward_tensors(
            x,
            data.edge_index,
            edge_attr,
            batch,
            group_probs,
        )

    def forward_tensors(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        group_probs: torch.Tensor,
    ):
        probs_expanded = group_probs[batch]
        node_out = self.input_proj(torch.cat([x, probs_expanded], dim=1))
        for block in self.blocks:
            node_out = block(node_out, edge_index, edge_attr)
        return self.node_head(node_out)

    @torch.jit.ignore
    def extract_embeddings(self, data: Data) -> torch.Tensor:
        x = self.node_features(data)
        edge_attr = self.edge_features(data)
        batch = self.node_graph_id(data)
        edge_index = data.edge_index
        group_probs = self._graph_features(data, batch=batch, dtype=x.dtype, device=x.device)
        probs_expanded = group_probs[batch]
        x = self.input_proj(torch.cat([x, probs_expanded], dim=1))

        for block in self.blocks:
            x = block(x, edge_index, edge_attr)
        return x

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
                group_probs: torch.Tensor,
            ):
                return self.model.forward_tensors(
                    x,
                    edge_index,
                    edge_attr,
                    batch,
                    group_probs,
                )

        scripted = torch.jit.script(_Scriptable(self))
        if path is not None:
            scripted.save(str(path))
        return scripted
