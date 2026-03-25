from __future__ import annotations

from typing import Any

import torch
from torch_geometric.data import Data

from ...config import GraphTensorDims
from ..structured_loader import StructuredLoader


class GraphLoader(StructuredLoader):
    """Structured loader for graph batches with overridable chunk slicing."""

    EDGE_TEMPLATE_CACHE_ENABLED = False
    EDGE_TEMPLATE_CACHE_MAX_ENTRIES: int | None = None

    def graph_dims_or_default(self) -> GraphTensorDims:
        dims = getattr(self, "graph_dims", None)
        if isinstance(dims, GraphTensorDims):
            return dims
        return GraphTensorDims(node_feature_dim=0, edge_feature_dim=0)

    def data_struct_fields(self) -> tuple[str, ...]:
        """Declared graph-layer batch contract."""
        return (
            "x_node",
            "x_edge",
            "x_graph",
            "y_node",
            "y_edge",
            "y_graph",
            "edge_index",
            "node_graph_id",
            "edge_graph_id",
            "graph_ptr",
            "node_ptr",
            "edge_ptr",
        )

    def empty_node_feature_dim(self) -> int:
        return max(0, int(self.graph_dims_or_default().node_feature_dim))

    def empty_edge_feature_dim(self) -> int:
        return max(0, int(self.graph_dims_or_default().edge_feature_dim))

    def empty_graph_feature_dim(self) -> int:
        return max(0, int(self.graph_dims_or_default().graph_feature_dim))

    def empty_node_target_dim(self) -> int:
        return max(0, int(self.graph_dims_or_default().node_target_dim))

    def empty_edge_target_dim(self) -> int:
        return max(0, int(self.graph_dims_or_default().edge_target_dim))

    def empty_graph_target_dim(self) -> int:
        return max(0, int(self.graph_dims_or_default().graph_target_dim))

    def empty_data(self) -> Data:
        node_dim = self.empty_node_feature_dim()
        edge_dim = self.empty_edge_feature_dim()
        graph_dim = self.empty_graph_feature_dim()
        y_node_dim = self.empty_node_target_dim()
        y_edge_dim = self.empty_edge_target_dim()
        y_graph_dim = self.empty_graph_target_dim()
        data = Data(
            x_node=torch.empty((0, node_dim), dtype=torch.float32),
            x_edge=torch.empty((0, edge_dim), dtype=torch.float32),
            x_graph=torch.empty((0, graph_dim), dtype=torch.float32),
            y_node=torch.empty((0, y_node_dim), dtype=torch.float32),
            y_edge=torch.empty((0, y_edge_dim), dtype=torch.float32),
            y_graph=torch.empty((0, y_graph_dim), dtype=torch.float32),
            node_graph_id=torch.empty((0,), dtype=torch.int64),
            edge_index=torch.empty((2, 0), dtype=torch.int64),
            edge_graph_id=torch.empty((0,), dtype=torch.int64),
            graph_ptr=torch.empty((0,), dtype=torch.int64),
            node_ptr=torch.empty((0,), dtype=torch.int64),
            edge_ptr=torch.empty((0,), dtype=torch.int64),
            num_graphs=0,
            num_rows=0,
        )
        return data

    def _slice_chunk_batch(self, chunk: dict, g0: int, g1: int) -> Data:
        node_ptr = chunk["node_ptr"]
        edge_ptr = chunk["edge_ptr"]
        n0 = int(node_ptr[g0].item())
        n1 = int(node_ptr[g1].item())
        e0 = int(edge_ptr[g0].item())
        e1 = int(edge_ptr[g1].item())
        num_graphs = int(g1 - g0)
        graph_dim = self.empty_graph_feature_dim()
        y_node_dim = self.empty_node_target_dim()
        y_edge_dim = self.empty_edge_target_dim()
        y_graph_dim = self.empty_graph_target_dim()
        local_edge_index = chunk["edge_index"][:, e0:e1] - n0
        local_counts = (node_ptr[g0 + 1 : g1 + 1] - node_ptr[g0:g1]).to(dtype=torch.int64)
        node_graph_id = torch.repeat_interleave(torch.arange(num_graphs, dtype=torch.int64), local_counts)
        x_graph = (
            chunk["x_graph"][g0:g1]
            if ("x_graph" in chunk and chunk["x_graph"] is not None)
            else torch.empty((num_graphs, graph_dim), dtype=torch.float32)
        )
        y_node = (
            chunk["y_node"][n0:n1]
            if ("y_node" in chunk and chunk["y_node"] is not None)
            else torch.empty((n1 - n0, y_node_dim), dtype=torch.float32)
        )
        y_edge = (
            chunk["y_edge"][e0:e1]
            if ("y_edge" in chunk and chunk["y_edge"] is not None)
            else torch.empty((e1 - e0, y_edge_dim), dtype=torch.float32)
        )
        y_graph = (
            chunk["y_graph"][g0:g1]
            if ("y_graph" in chunk and chunk["y_graph"] is not None)
            else torch.empty((num_graphs, y_graph_dim), dtype=torch.float32)
        )
        graph_meta = {}
        if "graph_event_id" in chunk and chunk["graph_event_id"] is not None:
            graph_meta["graph_event_id"] = chunk["graph_event_id"][g0:g1]
        if "graph_time_group_id" in chunk and chunk["graph_time_group_id"] is not None:
            graph_meta["graph_time_group_id"] = chunk["graph_time_group_id"][g0:g1]
        edge_graph_id = (
            node_graph_id[local_edge_index[0]]
            if local_edge_index.numel() > 0
            else torch.empty((0,), dtype=torch.int64)
        )

        d = Data(
            x_node=chunk["x_node"][n0:n1],
            x_edge=chunk["x_edge"][e0:e1],
            x_graph=x_graph,
            node_graph_id=node_graph_id,
            edge_index=local_edge_index,
            edge_graph_id=edge_graph_id,
            node_ptr=(node_ptr[g0 : g1 + 1] - n0).to(dtype=torch.int64),
            edge_ptr=(edge_ptr[g0 : g1 + 1] - e0).to(dtype=torch.int64),
            graph_ptr=torch.tensor([0, num_graphs], dtype=torch.int64),
            y_edge=y_edge,
            y_node=y_node,
            y_graph=y_graph,
            num_graphs=num_graphs,
            **graph_meta,
        )
        return d

    def build_inference_model_input(
        self,
        *,
        batch,
        device: torch.device,
        cfg: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        _ = cfg
        x = batch.x_node.to(device, non_blocking=(device.type == "cuda"))
        edge_index = batch.edge_index.to(device, non_blocking=(device.type == "cuda"))
        edge_attr = batch.x_edge.to(device, non_blocking=(device.type == "cuda"))
        node_graph_id = batch.node_graph_id.to(device, non_blocking=(device.type == "cuda"))
        return (x, edge_index, edge_attr, node_graph_id), {}
