from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pioneerml.common.pipeline.services import BasePipelineService


class EventSplitterInferenceServiceBase(BasePipelineService):
    @staticmethod
    def resolve_paths(parquet_paths: list[str]) -> list[str]:
        resolved = [str(Path(p).expanduser().resolve()) for p in parquet_paths]
        if not resolved:
            raise RuntimeError("No parquet paths provided for inference.")
        return resolved

    @staticmethod
    def resolve_optional_paths(parquet_paths: list[str] | None) -> list[str] | None:
        if parquet_paths is None:
            return None
        return [str(Path(p).expanduser().resolve()) for p in parquet_paths]

    @staticmethod
    def count_input_rows(parquet_paths: list[str]) -> int:
        total = 0
        for p in parquet_paths:
            total += int(pq.ParquetFile(p).metadata.num_rows)
        return total

    @staticmethod
    def resolve_inference_runtime(config_json: dict) -> tuple[str, bool, bool, bool, int, int, int]:
        mode = str(config_json.get("mode", "inference")).strip().lower()
        if mode not in {"inference", "train"}:
            raise ValueError(f"Unsupported loader mode: {mode}. Expected 'inference' or 'train'.")
        use_group_probs = bool(config_json.get("use_group_probs", True))
        use_splitter_probs = bool(config_json.get("use_splitter_probs", True))
        use_endpoint_preds = bool(config_json.get("use_endpoint_preds", True))
        batch_size = max(1, int(config_json.get("batch_size", 8)))
        row_groups_per_chunk = max(
            1, int(config_json.get("chunk_row_groups", config_json.get("row_groups_per_chunk", 4)))
        )
        num_workers = max(0, int(config_json.get("chunk_workers", config_json.get("num_workers", 0))))
        return (
            mode,
            use_group_probs,
            use_splitter_probs,
            use_endpoint_preds,
            batch_size,
            row_groups_per_chunk,
            num_workers,
        )

    @staticmethod
    def build_prediction_table(
        *,
        node_event_ids_np: np.ndarray,
        node_local_idx_np: np.ndarray,
        node_time_group_ids_np: np.ndarray,
        edge_event_ids_np: np.ndarray,
        edge_src_local_np: np.ndarray,
        edge_dst_local_np: np.ndarray,
        edge_probs_np: np.ndarray,
        num_rows: int,
    ) -> pa.Table:
        if num_rows <= 0:
            num_rows = int(node_event_ids_np.max()) + 1 if node_event_ids_np.size > 0 else 0

        valid_nodes = (
            (node_event_ids_np >= 0)
            & (node_event_ids_np < num_rows)
            & (node_local_idx_np >= 0)
            & (node_time_group_ids_np >= 0)
        )
        ne = node_event_ids_np[valid_nodes]
        nl = node_local_idx_np[valid_nodes]
        ntg = node_time_group_ids_np[valid_nodes]
        if ne.size > 0:
            node_order = np.lexsort((nl, ne))
            ne = ne[node_order]
            ntg = ntg[node_order]
        node_counts = np.bincount(ne, minlength=num_rows).astype(np.int64, copy=False)
        node_offsets = np.zeros((num_rows + 1,), dtype=np.int64)
        node_offsets[1:] = np.cumsum(node_counts, dtype=np.int64)

        valid_edges = (
            (edge_event_ids_np >= 0)
            & (edge_event_ids_np < num_rows)
            & (edge_src_local_np >= 0)
            & (edge_dst_local_np >= 0)
        )
        ee = edge_event_ids_np[valid_edges]
        es = edge_src_local_np[valid_edges]
        ed = edge_dst_local_np[valid_edges]
        ep = edge_probs_np[valid_edges]
        if ee.size > 0:
            edge_order = np.lexsort((ed, es, ee))
            ee = ee[edge_order]
            es = es[edge_order]
            ed = ed[edge_order]
            ep = ep[edge_order]
        edge_counts = np.bincount(ee, minlength=num_rows).astype(np.int64, copy=False)
        edge_offsets = np.zeros((num_rows + 1,), dtype=np.int64)
        edge_offsets[1:] = np.cumsum(edge_counts, dtype=np.int64)

        return pa.table(
            {
                "event_id": pa.array(np.arange(num_rows, dtype=np.int64)),
                "time_group_ids": pa.ListArray.from_arrays(node_offsets, pa.array(ntg, type=pa.int64())),
                "edge_src_index": pa.ListArray.from_arrays(edge_offsets, pa.array(es, type=pa.int64())),
                "edge_dst_index": pa.ListArray.from_arrays(edge_offsets, pa.array(ed, type=pa.int64())),
                "pred_edge_affinity": pa.ListArray.from_arrays(edge_offsets, pa.array(ep, type=pa.float32())),
            }
        )
