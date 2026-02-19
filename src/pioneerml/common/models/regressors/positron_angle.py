"""Positron momentum-vector quantile regressor."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.data import Data

from pioneerml.common.models.blocks import FullGraphTransformerBlock
from pioneerml.common.models.components.quantile_output_head import QuantileOutputHead


class PositronAngleModel(nn.Module):
    """Predicts positron `(px, py, pz)` quantiles per time group."""

    def __init__(
        self,
        in_channels: int = 4,
        group_prob_dimension: int = 3,
        splitter_prob_dimension: int = 3,
        endpoint_pred_dimension: int = 18,
        event_affinity_dimension: int = 3,
        pion_stop_pred_dimension: int = 3,
        hidden: int = 192,
        heads: int = 4,
        layers: int = 3,
        dropout: float = 0.1,
        output_dim: int = 9,
        quantiles: tuple[float, ...] = (0.16, 0.50, 0.84),
    ):
        super().__init__()
        self.group_prob_dimension = int(group_prob_dimension)
        self.splitter_prob_dimension = int(splitter_prob_dimension)
        self.endpoint_pred_dimension = int(endpoint_pred_dimension)
        self.event_affinity_dimension = int(event_affinity_dimension)
        self.pion_stop_pred_dimension = int(pion_stop_pred_dimension)
        self.num_outputs = 3
        self.quantiles = tuple(float(q) for q in quantiles)

        expected_output_dim = int(self.num_outputs * len(self.quantiles))
        if int(output_dim) != expected_output_dim:
            raise ValueError(
                f"output_dim ({output_dim}) must equal num_outputs*num_quantiles ({expected_output_dim})."
            )
        self.output_dim = expected_output_dim

        input_dim = (
            int(in_channels)
            + self.group_prob_dimension
            + self.splitter_prob_dimension
            + self.endpoint_pred_dimension
            + self.event_affinity_dimension
            + self.pion_stop_pred_dimension
        )
        self.input_proj = nn.Linear(input_dim, hidden)

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

        pooled_dim = (
            hidden * 3
            + self.group_prob_dimension
            + self.endpoint_pred_dimension
            + self.event_affinity_dimension
            + self.pion_stop_pred_dimension
            + 2
        )
        self.head = nn.Sequential(
            nn.Linear(pooled_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        self.quantile_head = QuantileOutputHead(
            input_dim=hidden // 2,
            num_points=1,
            coords=self.num_outputs,
            quantiles=list(self.quantiles),
        )

    @torch.jit.ignore
    def forward(self, data: Data) -> torch.Tensor:
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=data.x.device)

        num_graphs = int(getattr(data, "num_graphs", 0))
        if num_graphs <= 0 and batch.numel() > 0:
            num_graphs = int(batch.max().item()) + 1

        group_probs = getattr(data, "group_probs", None)
        if group_probs is None:
            group_probs = torch.zeros(
                (num_graphs, self.group_prob_dimension),
                dtype=data.x.dtype,
                device=data.x.device,
            )

        splitter_probs = getattr(data, "splitter_probs", None)
        if splitter_probs is None:
            splitter_probs = torch.zeros(
                (data.x.shape[0], self.splitter_prob_dimension),
                dtype=data.x.dtype,
                device=data.x.device,
            )

        endpoint_preds = getattr(data, "endpoint_preds", None)
        if endpoint_preds is None:
            endpoint_preds = torch.zeros(
                (num_graphs, self.endpoint_pred_dimension),
                dtype=data.x.dtype,
                device=data.x.device,
            )
        pion_stop_preds = getattr(data, "pion_stop_preds", None)
        if pion_stop_preds is None:
            pion_stop_preds = torch.zeros(
                (num_graphs, self.pion_stop_pred_dimension),
                dtype=data.x.dtype,
                device=data.x.device,
            )

        event_affinity = getattr(data, "event_affinity", None)
        if event_affinity is None:
            event_affinity = torch.zeros(
                (num_graphs, self.event_affinity_dimension),
                dtype=data.x.dtype,
                device=data.x.device,
            )

        return self.forward_tensors(
            data.x,
            data.edge_index,
            data.edge_attr,
            batch,
            group_probs,
            splitter_probs,
            endpoint_preds,
            event_affinity,
            pion_stop_preds,
            num_graphs=num_graphs,
        )

    @staticmethod
    def _match_graph_feature_rows(
        feature: torch.Tensor,
        *,
        num_rows: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if num_rows <= 0:
            return torch.zeros((0, width), dtype=dtype, device=device)
        if feature is None or feature.numel() == 0:
            return torch.zeros((num_rows, width), dtype=dtype, device=device)
        if feature.dim() != 2 or int(feature.shape[1]) != int(width):
            return torch.zeros((num_rows, width), dtype=dtype, device=device)
        if feature.shape[0] == num_rows:
            return feature.to(dtype=dtype, device=device)
        if feature.shape[0] > num_rows:
            return feature[:num_rows].to(dtype=dtype, device=device)
        out = torch.zeros((num_rows, width), dtype=dtype, device=device)
        out[: feature.shape[0]] = feature.to(dtype=dtype, device=device)
        return out

    @staticmethod
    def _match_node_feature_rows(
        feature: torch.Tensor,
        *,
        num_rows: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if num_rows <= 0:
            return torch.zeros((0, width), dtype=dtype, device=device)
        if feature is None or feature.numel() == 0:
            return torch.zeros((num_rows, width), dtype=dtype, device=device)
        if feature.dim() != 2 or int(feature.shape[1]) != int(width):
            return torch.zeros((num_rows, width), dtype=dtype, device=device)
        if feature.shape[0] == num_rows:
            return feature.to(dtype=dtype, device=device)
        if feature.shape[0] > num_rows:
            return feature[:num_rows].to(dtype=dtype, device=device)
        out = torch.zeros((num_rows, width), dtype=dtype, device=device)
        out[: feature.shape[0]] = feature.to(dtype=dtype, device=device)
        return out

    def forward_tensors(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        group_probs: torch.Tensor,
        splitter_probs: torch.Tensor,
        endpoint_preds: torch.Tensor,
        event_affinity: torch.Tensor,
        pion_stop_preds: torch.Tensor,
        num_graphs: int = 0,
    ) -> torch.Tensor:
        if num_graphs <= 0 and batch.numel() > 0:
            num_graphs = int(batch.max().item()) + 1

        raw_x = x
        num_nodes = int(x.shape[0])
        device = x.device
        dtype = x.dtype

        group_probs = self._match_graph_feature_rows(
            group_probs,
            num_rows=num_graphs,
            width=self.group_prob_dimension,
            dtype=dtype,
            device=device,
        )
        endpoint_preds = self._match_graph_feature_rows(
            endpoint_preds,
            num_rows=num_graphs,
            width=self.endpoint_pred_dimension,
            dtype=dtype,
            device=device,
        )
        event_affinity = self._match_graph_feature_rows(
            event_affinity,
            num_rows=num_graphs,
            width=self.event_affinity_dimension,
            dtype=dtype,
            device=device,
        )
        pion_stop_preds = self._match_graph_feature_rows(
            pion_stop_preds,
            num_rows=num_graphs,
            width=self.pion_stop_pred_dimension,
            dtype=dtype,
            device=device,
        )
        splitter_probs = self._match_node_feature_rows(
            splitter_probs,
            num_rows=num_nodes,
            width=self.splitter_prob_dimension,
            dtype=dtype,
            device=device,
        )

        if num_graphs > 0 and num_nodes > 0:
            expanded_group_probs = group_probs[batch]
            expanded_endpoint = endpoint_preds[batch]
            expanded_event_affinity = event_affinity[batch]
            expanded_pion_stop = pion_stop_preds[batch]
        else:
            expanded_group_probs = torch.zeros((num_nodes, self.group_prob_dimension), dtype=dtype, device=device)
            expanded_endpoint = torch.zeros((num_nodes, self.endpoint_pred_dimension), dtype=dtype, device=device)
            expanded_event_affinity = torch.zeros((num_nodes, self.event_affinity_dimension), dtype=dtype, device=device)
            expanded_pion_stop = torch.zeros((num_nodes, self.pion_stop_pred_dimension), dtype=dtype, device=device)

        x = self.input_proj(
            torch.cat(
                [
                    raw_x,
                    expanded_group_probs,
                    splitter_probs,
                    expanded_endpoint,
                    expanded_event_affinity,
                    expanded_pion_stop,
                ],
                dim=1,
            )
        )

        for block in self.blocks:
            x = block(x, edge_index, edge_attr)

        feat_dim = int(x.size(1))
        pool_x = x.new_zeros((num_graphs, feat_dim))
        pool_y = x.new_zeros((num_graphs, feat_dim))
        pool_all = x.new_zeros((num_graphs, feat_dim))
        counts_x = x.new_zeros((num_graphs, 1))
        counts_y = x.new_zeros((num_graphs, 1))
        counts_all = x.new_zeros((num_graphs, 1))

        if num_nodes > 0 and num_graphs > 0:
            one_vec = x.new_ones((num_nodes, 1))
            counts_all.index_add_(0, batch, one_vec)
            pool_all.index_add_(0, batch, x)

            raw_view = raw_x[:, 3].to(torch.long)
            mask_x = raw_view == 0
            mask_y = raw_view == 1

            bid_x = batch[mask_x]
            x_x = x[mask_x]
            pool_x.index_add_(0, bid_x, x_x)
            counts_x.index_add_(0, bid_x, x_x.new_ones((x_x.size(0), 1)))

            bid_y = batch[mask_y]
            x_y = x[mask_y]
            pool_y.index_add_(0, bid_y, x_y)
            counts_y.index_add_(0, bid_y, x_y.new_ones((x_y.size(0), 1)))

        pool_all = pool_all / counts_all.clamp_min(1.0)
        pool_x = pool_x / counts_x.clamp_min(1.0)
        pool_y = pool_y / counts_y.clamp_min(1.0)

        has_x = (counts_x > 0).to(x.dtype)
        has_y = (counts_y > 0).to(x.dtype)

        out = torch.cat(
            [pool_x, pool_y, pool_all, group_probs, endpoint_preds, event_affinity, pion_stop_preds, has_x, has_y],
            dim=1,
        )
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
            def __init__(self, model: PositronAngleModel):
                super().__init__()
                self.model = model

            def forward(
                self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                batch: torch.Tensor,
                group_probs: torch.Tensor,
                splitter_probs: torch.Tensor,
                endpoint_preds: torch.Tensor,
                event_affinity: torch.Tensor,
                pion_stop_preds: torch.Tensor,
            ) -> torch.Tensor:
                return self.model.forward_tensors(
                    x,
                    edge_index,
                    edge_attr,
                    batch,
                    group_probs,
                    splitter_probs,
                    endpoint_preds,
                    event_affinity,
                    pion_stop_preds,
                )

        scripted = torch.jit.script(_Scriptable(self))
        if path is not None:
            scripted.save(str(path))
        return scripted
