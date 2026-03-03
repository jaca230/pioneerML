"""Time-group endpoint regressor for per-group start/end coordinates."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.data import Data

from pioneerml.common.models.graph.transformer.regressors.base_graph_regressor_model import BaseGraphRegressorModel
from pioneerml.common.models.components.quantile_output_head import QuantileOutputHead


class EndpointRegressor(BaseGraphRegressorModel):
    """Predicts endpoint quantiles per time-group graph as `[2 points, 3 coords, 3 quantiles]`."""

    def __init__(
        self,
        node_dim: int = 4,
        graph_dim: int = 3,
        splitter_prob_dimension: int = 3,
        edge_dim: int = 4,
        hidden: int = 192,
        heads: int = 4,
        layers: int = 3,
        dropout: float = 0.1,
        output_dim: int = 18,
        quantiles: tuple[float, ...] = (0.16, 0.50, 0.84),
    ):
        super().__init__(node_dim=node_dim, edge_dim=edge_dim, graph_dim=graph_dim, hidden=hidden, dropout=dropout)
        self.group_prob_dimension = int(graph_dim)
        self.splitter_prob_dimension = int(splitter_prob_dimension)
        self.num_points = 2
        self.num_coords = 3
        self.quantiles = tuple(float(q) for q in quantiles)
        expected_output_dim = int(self.num_points * self.num_coords * len(self.quantiles))
        if int(output_dim) != expected_output_dim:
            raise ValueError(
                f"output_dim ({output_dim}) must equal num_points*coords*num_quantiles ({expected_output_dim})."
            )
        self.output_dim = expected_output_dim

        input_dim = int(node_dim) + self.group_prob_dimension + self.splitter_prob_dimension
        self.input_proj = nn.Linear(input_dim, hidden)

        self.blocks = self.build_transformer_blocks(
            hidden_dim=hidden,
            num_layers=layers,
            num_heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
        )

        pooled_dim = hidden * 3 + self.group_prob_dimension + 3
        self.head = self.build_mlp_head(
            input_dim=pooled_dim,
            hidden_dims=(hidden, hidden // 2),
            output_dim=hidden // 2,
            dropout=dropout,
        )
        self.quantile_head = QuantileOutputHead(
            input_dim=hidden // 2,
            num_points=self.num_points,
            coords=self.num_coords,
            quantiles=list(self.quantiles),
        )

    @torch.jit.ignore
    def forward(self, data: Data) -> torch.Tensor:
        node_features = self.node_features(data)
        edge_index = data.edge_index
        edge_features = self.edge_features(data)
        node_graph_id = self.node_graph_id(data)

        num_graphs = int(node_graph_id.max().item()) + 1 if node_graph_id.numel() > 0 else 0

        u = getattr(data, "u", None)
        if u is None:
            u = torch.zeros((num_graphs, 1), dtype=node_features.dtype, device=node_features.device)

        group_probs = getattr(data, "group_probs", None)
        if group_probs is None:
            graph_features = self.graph_features(data)
            if graph_features.dim() == 2 and int(graph_features.shape[1]) >= self.group_prob_dimension:
                group_probs = graph_features[:, : self.group_prob_dimension]
        if group_probs is None:
            group_probs = torch.zeros(
                (num_graphs, self.group_prob_dimension),
                dtype=node_features.dtype,
                device=node_features.device,
            )

        splitter_probs = getattr(data, "splitter_probs", None)
        if splitter_probs is None:
            splitter_probs = torch.zeros(
                (node_features.shape[0], self.splitter_prob_dimension),
                dtype=node_features.dtype,
                device=node_features.device,
            )

        return self.forward_tensors(
            node_features,
            edge_index,
            edge_features,
            node_graph_id,
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
        out = self.head(out)
        out = self.quantile_head(out)
        return out.reshape(out.size(0), self.output_dim)

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
