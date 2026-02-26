from __future__ import annotations

import torch
from torch_geometric.data import Data

from ..structured_loader import StructuredLoader


class GraphLoader(StructuredLoader):
    """Structured loader for graph batches with overridable chunk slicing."""

    def _empty_node_feature_dim(self) -> int:
        return max(0, int(getattr(self, "node_feature_dim", 0)))

    def _empty_edge_feature_dim(self) -> int:
        return max(0, int(getattr(self, "edge_feature_dim", 0)))

    def _empty_target_dim(self) -> int:
        return max(0, int(getattr(self, "num_classes", getattr(self, "target_dim", 0))))

    def _include_time_group_ids_in_empty_data(self) -> bool:
        return False

    def empty_data(self) -> tuple[Data, torch.Tensor]:
        node_dim = self._empty_node_feature_dim()
        edge_dim = self._empty_edge_feature_dim()
        target_dim = self._empty_target_dim()
        data = Data(
            x=torch.empty((0, node_dim), dtype=torch.float32),
            edge_index=torch.empty((2, 0), dtype=torch.int64),
            edge_attr=torch.empty((0, edge_dim), dtype=torch.float32),
        )
        if self._include_time_group_ids_in_empty_data():
            data.time_group_ids = torch.empty((0,), dtype=torch.int64)

        targets = torch.empty((0, target_dim), dtype=torch.float32)
        data.num_graphs = 0
        data.num_rows = 0
        if bool(getattr(self, "include_targets", False)):
            data.y = targets
        return data, targets

    def _slice_chunk_batch(self, chunk: dict, g0: int, g1: int) -> Data:
        node_ptr = chunk["node_ptr"]
        edge_ptr = chunk["edge_ptr"]
        n0 = int(node_ptr[g0].item())
        n1 = int(node_ptr[g1].item())
        e0 = int(edge_ptr[g0].item())
        e1 = int(edge_ptr[g1].item())

        d = Data(
            x=chunk["x"][n0:n1],
            edge_index=(chunk["edge_index"][:, e0:e1] - n0),
            edge_attr=chunk["edge_attr"][e0:e1],
        )
        if "targets" in chunk:
            d.y = chunk["targets"][g0:g1]
        local_counts = (node_ptr[g0 + 1 : g1 + 1] - node_ptr[g0:g1]).to(dtype=torch.int64)
        d.batch = torch.repeat_interleave(torch.arange(g1 - g0, dtype=torch.int64), local_counts)
        d.num_graphs = int(g1 - g0)
        return d
