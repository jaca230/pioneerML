from __future__ import annotations

import numpy as np
import pyarrow as pa
import torch
from torch_geometric.data import Data

from .time_group_graph_loader import TimeGroupGraphLoader


class GroupClassifierGraphLoader(TimeGroupGraphLoader):
    """Chunked graph loader for group-classifier training and inference."""

    GRAPH_COLUMNS = [
        "event_id",
        "hits_time_group",
        "hits_strip_type",
        "hits_coord",
        "hits_z",
        "hits_edep",
    ]
    TARGET_COLUMNS = [
        "group_pion_in",
        "group_muon_in",
        "group_mip_in",
    ]
    NODE_FEATURE_DIM = 4
    EDGE_FEATURE_DIM = 4
    NUM_CLASSES = 3
    MODE_TRAIN = "train"
    MODE_INFERENCE = "inference"

    def __init__(
        self,
        parquet_paths: list[str],
        *,
        mode: str = MODE_TRAIN,
        batch_size: int = 64,
        row_groups_per_chunk: int = 4,
        num_workers: int = 0,
        columns: list[str] | None = None,
    ) -> None:
        mode_norm = str(mode).strip().lower()
        if mode_norm not in (self.MODE_TRAIN, self.MODE_INFERENCE):
            raise ValueError(f"Unsupported mode: {mode}. Expected 'train' or 'inference'.")
        self.mode = mode_norm

        required = self._required_columns_for_mode(self.mode)
        resolved_columns = list(columns) if columns is not None else required
        super().__init__(
            parquet_paths=parquet_paths,
            batch_size=batch_size,
            row_groups_per_chunk=row_groups_per_chunk,
            num_workers=num_workers,
            columns=resolved_columns,
        )

        missing = [c for c in required if c not in self.columns]
        if missing:
            raise ValueError(f"Missing required columns for mode={self.mode}: {missing}")

    @staticmethod
    def _required_columns_for_mode(mode: str) -> list[str]:
        if mode == GroupClassifierGraphLoader.MODE_TRAIN:
            return [*GroupClassifierGraphLoader.GRAPH_COLUMNS, *GroupClassifierGraphLoader.TARGET_COLUMNS]
        return list(GroupClassifierGraphLoader.GRAPH_COLUMNS)

    def _constructor_kwargs(self) -> dict[str, object]:
        kwargs = super()._constructor_kwargs()
        kwargs["mode"] = self.mode
        return kwargs

    @property
    def include_targets(self) -> bool:
        return self.mode == self.MODE_TRAIN

    def empty_data(self) -> tuple[Data, torch.Tensor]:
        data = Data(
            x=torch.empty((0, self.NODE_FEATURE_DIM), dtype=torch.float32),
            edge_index=torch.empty((2, 0), dtype=torch.int64),
            edge_attr=torch.empty((0, self.EDGE_FEATURE_DIM), dtype=torch.float32),
            time_group_ids=torch.empty((0,), dtype=torch.int64),
        )
        targets = torch.empty((0, self.NUM_CLASSES), dtype=torch.float32)
        data.num_graphs = 0
        data.num_groups = 0
        data.num_rows = 0
        if self.include_targets:
            data.y = targets
        return data, targets

    def _extract_chunk_inputs(self, table: pa.Table) -> dict[str, np.ndarray]:
        hits_time_group_offsets, hits_time_group_values = self._extract_list_column(table, "hits_time_group", np.int64)
        _, hits_coord_values = self._extract_list_column(table, "hits_coord", np.float32)
        _, hits_z_values = self._extract_list_column(table, "hits_z", np.float32)
        _, hits_edep_values = self._extract_list_column(table, "hits_edep", np.float32)
        _, hits_strip_type_values = self._extract_list_column(table, "hits_strip_type", np.int32)

        out: dict[str, np.ndarray] = {
            "event_ids": self._to_np(table.column("event_id").chunk(0), np.int64),
            "hits_time_group_offsets": hits_time_group_offsets,
            "hits_time_group_values": hits_time_group_values,
            "hits_coord_values": hits_coord_values,
            "hits_z_values": hits_z_values,
            "hits_edep_values": hits_edep_values,
            "hits_strip_type_values": hits_strip_type_values,
        }

        if self.include_targets:
            group_pion_in_offsets, group_pion_in_values = self._extract_list_column(table, "group_pion_in", np.int32)
            group_muon_in_offsets, group_muon_in_values = self._extract_list_column(table, "group_muon_in", np.int32)
            group_mip_in_offsets, group_mip_in_values = self._extract_list_column(table, "group_mip_in", np.int32)
            out.update(
                {
                    "group_pion_in_offsets": group_pion_in_offsets,
                    "group_pion_in_values": group_pion_in_values,
                    "group_muon_in_offsets": group_muon_in_offsets,
                    "group_muon_in_values": group_muon_in_values,
                    "group_mip_in_offsets": group_mip_in_offsets,
                    "group_mip_in_values": group_mip_in_values,
                }
            )
        return out

    @staticmethod
    def _fill_target_column_from_group_values(
        *,
        y_out: np.ndarray,
        dst_col: int,
        vals: np.ndarray,
        offs: np.ndarray,
        total_graphs: int,
        local_gid: np.ndarray,
        row_ids_graph: np.ndarray,
    ) -> None:
        if total_graphs == 0 or vals.size == 0:
            return
        counts = (offs[1:] - offs[:-1]).astype(np.int64, copy=False)
        valid = local_gid < counts[row_ids_graph]
        if not np.any(valid):
            return
        idx = offs[row_ids_graph[valid]] + local_gid[valid]
        y_out[valid, dst_col] = vals[idx].astype(np.float32, copy=False)

    @classmethod
    def _populate_target_labels(
        cls,
        *,
        y_out: np.ndarray,
        total_graphs: int,
        local_gid: np.ndarray,
        row_ids_graph: np.ndarray,
        group_pion_in_values: np.ndarray,
        group_pion_in_offsets: np.ndarray,
        group_muon_in_values: np.ndarray,
        group_muon_in_offsets: np.ndarray,
        group_mip_in_values: np.ndarray,
        group_mip_in_offsets: np.ndarray,
    ) -> None:
        cls._fill_target_column_from_group_values(
            y_out=y_out,
            dst_col=0,
            vals=group_pion_in_values,
            offs=group_pion_in_offsets,
            total_graphs=total_graphs,
            local_gid=local_gid,
            row_ids_graph=row_ids_graph,
        )
        cls._fill_target_column_from_group_values(
            y_out=y_out,
            dst_col=1,
            vals=group_muon_in_values,
            offs=group_muon_in_offsets,
            total_graphs=total_graphs,
            local_gid=local_gid,
            row_ids_graph=row_ids_graph,
        )
        cls._fill_target_column_from_group_values(
            y_out=y_out,
            dst_col=2,
            vals=group_mip_in_values,
            offs=group_mip_in_offsets,
            total_graphs=total_graphs,
            local_gid=local_gid,
            row_ids_graph=row_ids_graph,
        )

    @staticmethod
    def _populate_node_tensors(
        *,
        total_nodes: int,
        global_group_id: np.ndarray,
        hit_coord_values: np.ndarray,
        hit_z_values: np.ndarray,
        hit_edep_values: np.ndarray,
        hit_strip_type_values: np.ndarray,
        hit_time_group_values: np.ndarray,
        x_out: np.ndarray,
        time_group_ids_out: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if total_nodes > 0:
            order = np.argsort(global_group_id, kind="stable")
            x_out[:, 0] = hit_coord_values[order]
            x_out[:, 1] = hit_z_values[order]
            x_out[:, 2] = hit_edep_values[order]
            x_out[:, 3] = hit_strip_type_values[order].astype(np.float32, copy=False)
            time_group_ids_out[:] = hit_time_group_values[order]
            return x_out[:, 0], x_out[:, 1], x_out[:, 2], hit_strip_type_values[order]
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )

    def _populate_graph_edges(
        self,
        *,
        node_counts: np.ndarray,
        node_ptr: np.ndarray,
        edge_ptr: np.ndarray,
        sorted_coord_values: np.ndarray,
        sorted_z_values: np.ndarray,
        sorted_edep_values: np.ndarray,
        sorted_strip_type_values: np.ndarray,
        edge_index_out: np.ndarray,
        edge_attr_out: np.ndarray,
    ) -> None:
        edge_tpl_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for k in np.unique(node_counts):
            k = int(k)
            if k <= 1:
                continue
            src, dst = self._complete_digraph_cached(k, edge_tpl_cache)
            ecount = int(src.shape[0])
            graphs = np.flatnonzero(node_counts == k)
            if graphs.size == 0:
                continue
            rel_edge = np.arange(ecount, dtype=np.int64)
            graph_block = int(self.edge_populate_graph_block)
            for i in range(0, graphs.size, graph_block):
                g = graphs[i : i + graph_block]
                node_base = node_ptr[g]
                edge_base = edge_ptr[g]
                pos = (edge_base[:, None] + rel_edge[None, :]).reshape(-1)
                src_idx = (node_base[:, None] + src[None, :]).reshape(-1)
                dst_idx = (node_base[:, None] + dst[None, :]).reshape(-1)
                edge_index_out[0, pos] = src_idx
                edge_index_out[1, pos] = dst_idx
                edge_attr_out[pos, 0] = sorted_coord_values[dst_idx] - sorted_coord_values[src_idx]
                edge_attr_out[pos, 1] = sorted_z_values[dst_idx] - sorted_z_values[src_idx]
                edge_attr_out[pos, 2] = sorted_edep_values[dst_idx] - sorted_edep_values[src_idx]
                edge_attr_out[pos, 3] = (sorted_strip_type_values[src_idx] == sorted_strip_type_values[dst_idx]).astype(
                    np.float32
                )

    def _build_chunk_graph_arrays(self, *, table: pa.Table) -> dict:
        n_rows = int(table.num_rows)
        chunk_in = self._extract_chunk_inputs(table)

        target_count_candidates: list[np.ndarray] = []
        if self.include_targets:
            target_count_candidates = [
                self._counts_from_offsets(chunk_in["group_pion_in_offsets"]),
                self._counts_from_offsets(chunk_in["group_muon_in_offsets"]),
                self._counts_from_offsets(chunk_in["group_mip_in_offsets"]),
            ]

        layout = self._compute_group_layout(
            n_rows=n_rows,
            hits_time_group_offsets=chunk_in["hits_time_group_offsets"],
            hits_time_group_values=chunk_in["hits_time_group_values"],
            row_group_count_candidates=target_count_candidates,
        )

        total_nodes = int(layout["total_nodes"])
        total_edges = int(layout["total_edges"])
        total_graphs = int(layout["total_graphs"])
        node_ptr = layout["node_ptr"]
        edge_ptr = layout["edge_ptr"]
        node_counts = layout["node_counts"]
        global_group_id = layout["global_group_id"]
        row_group_counts = layout["row_group_counts"]
        row_group_base = layout["row_group_base"]

        x_out = np.empty((total_nodes, self.NODE_FEATURE_DIM), dtype=np.float32)
        edge_index_out = np.empty((2, total_edges), dtype=np.int64)
        edge_attr_out = np.empty((total_edges, self.EDGE_FEATURE_DIM), dtype=np.float32)
        tgroup_out = np.empty((total_nodes,), dtype=np.int64)
        row_ids_graph, local_gid, graph_event_ids, graph_group_ids = self._build_graph_index_mapping(
            total_graphs=total_graphs,
            n_rows=n_rows,
            row_group_counts=row_group_counts,
            row_group_base=row_group_base,
            event_ids=chunk_in["event_ids"],
        )

        coord_sorted, z_sorted, e_sorted, view_sorted = self._populate_node_tensors(
            total_nodes=total_nodes,
            global_group_id=global_group_id,
            hit_coord_values=chunk_in["hits_coord_values"],
            hit_z_values=chunk_in["hits_z_values"],
            hit_edep_values=chunk_in["hits_edep_values"],
            hit_strip_type_values=chunk_in["hits_strip_type_values"],
            hit_time_group_values=chunk_in["hits_time_group_values"],
            x_out=x_out,
            time_group_ids_out=tgroup_out,
        )

        targets_torch: torch.Tensor | None = None
        if self.include_targets:
            y_out = np.zeros((total_graphs, self.NUM_CLASSES), dtype=np.float32)
            self._populate_target_labels(
                y_out=y_out,
                total_graphs=total_graphs,
                local_gid=local_gid,
                row_ids_graph=row_ids_graph,
                group_pion_in_values=chunk_in["group_pion_in_values"],
                group_pion_in_offsets=chunk_in["group_pion_in_offsets"],
                group_muon_in_values=chunk_in["group_muon_in_values"],
                group_muon_in_offsets=chunk_in["group_muon_in_offsets"],
                group_mip_in_values=chunk_in["group_mip_in_values"],
                group_mip_in_offsets=chunk_in["group_mip_in_offsets"],
            )
            targets_torch = torch.from_numpy(y_out)

        self._populate_graph_edges(
            node_counts=node_counts,
            node_ptr=node_ptr,
            edge_ptr=edge_ptr,
            sorted_coord_values=coord_sorted,
            sorted_z_values=z_sorted,
            sorted_edep_values=e_sorted,
            sorted_strip_type_values=view_sorted,
            edge_index_out=edge_index_out,
            edge_attr_out=edge_attr_out,
        )

        chunk_out = {
            "x": torch.from_numpy(x_out),
            "edge_index": torch.from_numpy(edge_index_out),
            "edge_attr": torch.from_numpy(edge_attr_out),
            "time_group_ids": torch.from_numpy(tgroup_out),
            "graph_event_ids": torch.from_numpy(graph_event_ids),
            "graph_group_ids": torch.from_numpy(graph_group_ids),
            "node_ptr": torch.from_numpy(node_ptr),
            "edge_ptr": torch.from_numpy(edge_ptr),
            "num_rows": n_rows,
            "num_graphs": int(total_graphs),
        }
        if targets_torch is not None:
            chunk_out["targets"] = targets_torch
        return chunk_out

    @staticmethod
    def _slice_chunk_batch(chunk: dict, g0: int, g1: int) -> Data:
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
            time_group_ids=chunk["time_group_ids"][n0:n1],
        )
        if "targets" in chunk:
            d.y = chunk["targets"][g0:g1]
        local_counts = (node_ptr[g0 + 1 : g1 + 1] - node_ptr[g0:g1]).to(dtype=torch.int64)
        d.batch = torch.repeat_interleave(torch.arange(g1 - g0, dtype=torch.int64), local_counts)
        d.event_ids = chunk["graph_event_ids"][g0:g1]
        d.group_ids = chunk["graph_group_ids"][g0:g1]
        d.num_graphs = int(g1 - g0)
        d.num_groups = int(g1 - g0)
        return d
