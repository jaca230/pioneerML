from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pioneerml.common.pipeline.services import BasePipelineService


class GroupSplitterInferenceServiceBase(BasePipelineService):
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
    def resolve_inference_runtime(config_json: dict) -> tuple[str, bool, int, int, int]:
        mode = str(config_json.get("mode", "inference")).strip().lower()
        if mode not in {"inference", "train"}:
            raise ValueError(f"Unsupported loader mode: {mode}. Expected 'inference' or 'train'.")
        use_group_probs = bool(config_json.get("use_group_probs", True))
        batch_size = max(1, int(config_json.get("batch_size", 64)))
        row_groups_per_chunk = max(
            1, int(config_json.get("chunk_row_groups", config_json.get("row_groups_per_chunk", 4)))
        )
        num_workers = max(0, int(config_json.get("chunk_workers", config_json.get("num_workers", 0))))
        return mode, use_group_probs, batch_size, row_groups_per_chunk, num_workers

    @staticmethod
    def build_prediction_table(
        *,
        event_ids_np: np.ndarray,
        time_group_ids_np: np.ndarray,
        probs_np: np.ndarray,
        num_rows: int,
    ) -> pa.Table:
        if num_rows <= 0:
            num_rows = int(event_ids_np.max()) + 1 if event_ids_np.size > 0 else 0

        valid = (event_ids_np >= 0) & (event_ids_np < num_rows) & (time_group_ids_np >= 0)
        event_ids_v = event_ids_np[valid]
        tg_v = time_group_ids_np[valid]
        probs_v = probs_np[valid]

        if event_ids_v.size > 0:
            order = np.lexsort((tg_v, event_ids_v))
            event_sorted = event_ids_v[order]
            tg_sorted = tg_v[order]
            probs_sorted = probs_v[order]
        else:
            event_sorted = np.zeros((0,), dtype=np.int64)
            tg_sorted = np.zeros((0,), dtype=np.int64)
            probs_sorted = np.zeros((0, 3), dtype=np.float32)

        counts = np.bincount(event_sorted, minlength=num_rows).astype(np.int64, copy=False)
        offsets = np.zeros((num_rows + 1,), dtype=np.int64)
        offsets[1:] = np.cumsum(counts, dtype=np.int64)

        event_id_col = pa.array(np.arange(num_rows, dtype=np.int64))
        time_group_col = pa.ListArray.from_arrays(offsets, pa.array(tg_sorted, type=pa.int64()))
        pred_pion = pa.ListArray.from_arrays(offsets, pa.array(probs_sorted[:, 0], type=pa.float32()))
        pred_muon = pa.ListArray.from_arrays(offsets, pa.array(probs_sorted[:, 1], type=pa.float32()))
        pred_mip = pa.ListArray.from_arrays(offsets, pa.array(probs_sorted[:, 2], type=pa.float32()))
        return pa.table(
            {
                "event_id": event_id_col,
                "time_group_ids": time_group_col,
                "pred_hit_pion": pred_pion,
                "pred_hit_muon": pred_muon,
                "pred_hit_mip": pred_mip,
            }
        )
