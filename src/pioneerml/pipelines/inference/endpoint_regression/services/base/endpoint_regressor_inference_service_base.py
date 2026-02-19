from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pioneerml.common.pipeline.services import BasePipelineService


class EndpointRegressorInferenceServiceBase(BasePipelineService):
    QUANTILE_SUFFIXES = ("q16", "q50", "q84")

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
    def resolve_inference_runtime(config_json: dict) -> tuple[str, bool, bool, int, int, int]:
        mode = str(config_json.get("mode", "inference")).strip().lower()
        if mode not in {"inference", "train"}:
            raise ValueError(f"Unsupported loader mode: {mode}. Expected 'inference' or 'train'.")
        use_group_probs = bool(config_json.get("use_group_probs", True))
        use_splitter_probs = bool(config_json.get("use_splitter_probs", True))
        batch_size = max(1, int(config_json.get("batch_size", 64)))
        row_groups_per_chunk = max(
            1, int(config_json.get("chunk_row_groups", config_json.get("row_groups_per_chunk", 4)))
        )
        num_workers = max(0, int(config_json.get("chunk_workers", config_json.get("num_workers", 0))))
        return (
            mode,
            use_group_probs,
            use_splitter_probs,
            batch_size,
            row_groups_per_chunk,
            num_workers,
        )

    @staticmethod
    def build_prediction_table(
        *,
        event_ids_np: np.ndarray,
        group_ids_np: np.ndarray,
        preds_np: np.ndarray,
        num_rows: int,
    ) -> pa.Table:
        if preds_np.ndim != 2:
            raise ValueError(f"Expected predictions to be 2D [N, D], got shape {preds_np.shape}.")

        pred_dim = int(preds_np.shape[1]) if preds_np.size > 0 else int(preds_np.shape[1])
        if pred_dim not in (6, 18):
            raise ValueError(f"Unsupported endpoint prediction dimension {pred_dim}. Expected 6 or 18.")

        if num_rows <= 0:
            num_rows = int(event_ids_np.max()) + 1 if event_ids_np.size > 0 else 0

        valid = (event_ids_np >= 0) & (event_ids_np < num_rows) & (group_ids_np >= 0)
        event_ids_v = event_ids_np[valid]
        group_ids_v = group_ids_np[valid]
        preds_v = preds_np[valid]

        if event_ids_v.size > 0:
            order = np.lexsort((group_ids_v, event_ids_v))
            event_sorted = event_ids_v[order]
            group_sorted = group_ids_v[order]
            preds_sorted = preds_v[order]
        else:
            event_sorted = np.zeros((0,), dtype=np.int64)
            group_sorted = np.zeros((0,), dtype=np.int64)
            preds_sorted = np.zeros((0, pred_dim), dtype=np.float32)

        counts = np.bincount(event_sorted, minlength=num_rows).astype(np.int64, copy=False)
        offsets = np.zeros((num_rows + 1,), dtype=np.int64)
        offsets[1:] = np.cumsum(counts, dtype=np.int64)

        event_id_col = pa.array(np.arange(num_rows, dtype=np.int64))
        time_group_col = pa.ListArray.from_arrays(offsets, pa.array(group_sorted, type=pa.int64()))

        if pred_dim == 18:
            preds_quant = preds_sorted.reshape(-1, 2, 3, 3)
        else:
            preds_quant = np.repeat(preds_sorted.reshape(-1, 2, 3, 1), repeats=3, axis=3)

        out_cols: dict[str, pa.Array] = {
            "event_id": event_id_col,
            "time_group_ids": time_group_col,
        }
        point_names = ("start", "end")
        coord_names = ("x", "y", "z")
        for point_idx, point_name in enumerate(point_names):
            for coord_idx, coord_name in enumerate(coord_names):
                vals = preds_quant[:, point_idx, coord_idx, :].astype(np.float32, copy=False)
                out_cols[f"pred_group_{point_name}_{coord_name}"] = pa.ListArray.from_arrays(
                    offsets, pa.array(vals[:, 1], type=pa.float32())
                )
                for q_idx, q_suffix in enumerate(EndpointRegressorInferenceServiceBase.QUANTILE_SUFFIXES):
                    out_cols[f"pred_group_{point_name}_{coord_name}_{q_suffix}"] = pa.ListArray.from_arrays(
                        offsets, pa.array(vals[:, q_idx], type=pa.float32())
                    )

        return pa.table(out_cols)
