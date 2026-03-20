"""Affinity scorer for evaluating group consistency."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import JumpingKnowledge

from pioneerml.common.integration.pytorch.models.architectures.graph.transformer.classifiers.base_graph_classifier_model import BaseGraphClassifierModel
from pioneerml.common.integration.pytorch.models.architectures.factory.registry import REGISTRY as ARCHITECTURE_REGISTRY


@ARCHITECTURE_REGISTRY.register("group_affinity")
class GroupAffinityModel(BaseGraphClassifierModel):
    """
    Graph-level affinity scorer.
    """

    def __init__(
        self,
        node_dim: int = 4,
        edge_dim: int = 4,
        graph_dim: int = 1,
        hidden: int = 128,
        heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__(node_dim=node_dim, edge_dim=edge_dim, graph_dim=graph_dim, hidden=hidden, dropout=dropout)
        self.num_layers = num_layers
        self.heads = heads

        self.input_proj = nn.Linear(node_dim, hidden)

        self.layers = self.build_transformer_blocks(
            hidden_dim=hidden,
            num_layers=num_layers,
            num_heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
        )

        self.jk = JumpingKnowledge(mode="cat")
        jk_dim = hidden * num_layers

        self.pool = self.build_attentional_pool(feature_dim=jk_dim, gate_hidden_dim=hidden)

        self.head = self.build_mlp_head(
            input_dim=jk_dim + 1,
            hidden_dims=(hidden,),
            output_dim=1,
        )

    def forward(self, data: Data) -> torch.Tensor:
        node_features = self.input_proj(self.node_features(data))
        edge_features = self.edge_features(data)
        node_graph_id = self.node_graph_id(data)
        graph_features = self.graph_features(data)
        layer_outputs = []
        for block in self.layers:
            node_features = block(node_features, data.edge_index, edge_features)
            layer_outputs.append(node_features)
        combined_node_features = self.jk(layer_outputs)
        pooled_graph_features = self.pool(combined_node_features, node_graph_id)
        if int(graph_features.shape[1]) > 0:
            global_features = graph_features[:, :1]
        else:
            global_features = pooled_graph_features.new_zeros((pooled_graph_features.shape[0], 1))
        out = torch.cat([pooled_graph_features, global_features], dim=1)
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
            def __init__(self, model: GroupAffinityModel):
                super().__init__()
                self.model = model

            def forward(
                self,
                x_node: torch.Tensor,
                edge_index: torch.Tensor,
                x_edge: torch.Tensor,
                node_graph_id: torch.Tensor,
                x_graph: torch.Tensor,
            ) -> torch.Tensor:
                data = Data(
                    x_node=x_node,
                    edge_index=edge_index,
                    x_edge=x_edge,
                    node_graph_id=node_graph_id,
                    x_graph=x_graph,
                )
                return self.model(data)

        scripted = torch.jit.script(_Scriptable(self))
        if path is not None:
            scripted.save(str(path))
        return scripted
