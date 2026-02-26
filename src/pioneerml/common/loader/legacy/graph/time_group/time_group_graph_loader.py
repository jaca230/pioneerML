from __future__ import annotations

import numpy as np

from ..base_graph_loader import BaseGraphLoader


class TimeGroupGraphLoader(BaseGraphLoader):
    """Graph loader base with shared time-group layout utilities."""

    @staticmethod
    def _counts_from_offsets(offsets: np.ndarray) -> np.ndarray:
        return (offsets[1:] - offsets[:-1]).astype(np.int64, copy=False)

    @staticmethod
    def _compute_time_group_counts(
        *,
        n_rows: int,
        hits_time_group_offsets: np.ndarray,
        hits_time_group_values: np.ndarray,
        hit_counts: np.ndarray,
    ) -> np.ndarray:
        if hits_time_group_values.size == 0:
            return np.zeros((n_rows,), dtype=np.int64)
        starts = hits_time_group_offsets[:-1].astype(np.int64, copy=False)
        safe_starts = np.minimum(starts, hits_time_group_values.size - 1)
        row_tg_max = np.maximum.reduceat(hits_time_group_values, safe_starts).astype(np.int64, copy=False)
        row_tg_max[hit_counts == 0] = -1
        return row_tg_max + 1

    @classmethod
    def _compute_group_layout(
        cls,
        *,
        n_rows: int,
        hits_time_group_offsets: np.ndarray,
        hits_time_group_values: np.ndarray,
        row_group_count_candidates: list[np.ndarray],
    ) -> dict[str, np.ndarray | int]:
        hit_counts = cls._counts_from_offsets(hits_time_group_offsets)
        total_nodes = int(hit_counts.sum())
        row_ids_hit = np.repeat(np.arange(n_rows, dtype=np.int64), hit_counts)
        hits_time_group_counts = cls._compute_time_group_counts(
            n_rows=n_rows,
            hits_time_group_offsets=hits_time_group_offsets,
            hits_time_group_values=hits_time_group_values,
            hit_counts=hit_counts,
        )

        row_group_counts = np.maximum.reduce([hits_time_group_counts, *row_group_count_candidates]).astype(
            np.int64, copy=False
        )
        graph_offsets = np.zeros((n_rows + 1,), dtype=np.int64)
        graph_offsets[1:] = np.cumsum(row_group_counts)
        total_graphs = int(graph_offsets[-1])
        row_group_base = graph_offsets[:-1]

        if total_nodes > 0 and total_graphs > 0:
            global_group_id = row_group_base[row_ids_hit] + hits_time_group_values
            node_counts = np.bincount(global_group_id, minlength=total_graphs).astype(np.int64, copy=False)
        else:
            global_group_id = np.zeros((0,), dtype=np.int64)
            node_counts = np.zeros((total_graphs,), dtype=np.int64)

        edge_counts = node_counts * np.maximum(node_counts - 1, 0)
        total_edges = int(edge_counts.sum())
        node_ptr = np.zeros((total_graphs + 1,), dtype=np.int64)
        edge_ptr = np.zeros((total_graphs + 1,), dtype=np.int64)
        if total_graphs > 0:
            node_ptr[1:] = np.cumsum(node_counts)
            edge_ptr[1:] = np.cumsum(edge_counts)

        return {
            "hit_counts": hit_counts,
            "total_nodes": total_nodes,
            "row_ids_hit": row_ids_hit,
            "hits_time_group_counts": hits_time_group_counts,
            "row_group_counts": row_group_counts,
            "total_graphs": total_graphs,
            "row_group_base": row_group_base,
            "global_group_id": global_group_id,
            "node_counts": node_counts,
            "total_edges": total_edges,
            "node_ptr": node_ptr,
            "edge_ptr": edge_ptr,
        }

    @staticmethod
    def _build_graph_index_mapping(
        *,
        total_graphs: int,
        n_rows: int,
        row_group_counts: np.ndarray,
        row_group_base: np.ndarray,
        event_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        graph_event_ids = np.zeros((total_graphs,), dtype=np.int64)
        graph_group_ids = np.zeros((total_graphs,), dtype=np.int64)
        if total_graphs > 0:
            graph_ids = np.arange(total_graphs, dtype=np.int64)
            row_ids_graph = np.repeat(np.arange(n_rows, dtype=np.int64), row_group_counts)
            local_gid = graph_ids - row_group_base[row_ids_graph]
            graph_event_ids[:] = row_ids_graph
            graph_group_ids[:] = local_gid
        else:
            row_ids_graph = np.zeros((0,), dtype=np.int64)
            local_gid = np.zeros((0,), dtype=np.int64)
        return row_ids_graph, local_gid, graph_event_ids, graph_group_ids
