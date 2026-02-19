from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pioneerml.common.pipeline.services import BasePipelineService


class PionStopInferenceServiceBase(BasePipelineService):
    QUANTILE_SUFFIXES = ("q16", "q50", "q84")
    TARGET_AXES = ("x", "y", "z")

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
    def resolve_inference_runtime(config_json: dict) -> tuple[str, bool, bool, bool, bool, int, int, int]:
        mode = str(config_json.get("mode", "inference")).strip().lower()
        if mode not in {"inference", "train"}:
            raise ValueError(f"Unsupported loader mode: {mode}. Expected 'inference' or 'train'.")
        use_group_probs = bool(config_json.get("use_group_probs", True))
        use_splitter_probs = bool(config_json.get("use_splitter_probs", True))
        use_endpoint_preds = bool(config_json.get("use_endpoint_preds", True))
        use_event_splitter_affinity = bool(config_json.get("use_event_splitter_affinity", True))
        batch_size = max(1, int(config_json.get("batch_size", 64)))
        row_groups_per_chunk = max(
            1, int(config_json.get("chunk_row_groups", config_json.get("row_groups_per_chunk", 4)))
        )
        num_workers = max(0, int(config_json.get("chunk_workers", config_json.get("num_workers", 0))))
        return (
            mode,
            use_group_probs,
            use_splitter_probs,
            use_endpoint_preds,
            use_event_splitter_affinity,
            batch_size,
            row_groups_per_chunk,
            num_workers,
        )

    @classmethod
    def build_prediction_table(
        cls,
        *,
        event_ids_np: np.ndarray,
        group_ids_np: np.ndarray,
        preds_np: np.ndarray,
        num_rows: int,
        source_parquet_path: str | None = None,
    ) -> pa.Table:
        if preds_np.ndim != 2:
            raise ValueError(f"Expected predictions to be 2D [N, D], got shape {preds_np.shape}.")

        pred_dim = int(preds_np.shape[1]) if preds_np.size > 0 else int(preds_np.shape[1])
        if pred_dim not in (3, 9):
            raise ValueError(f"Unsupported pion-stop prediction dimension {pred_dim}. Expected 3 or 9.")

        if num_rows <= 0:
            num_rows = int(event_ids_np.max()) + 1 if event_ids_np.size > 0 else 0

        if source_parquet_path is not None:
            main_table = pq.read_table(source_parquet_path, columns=["hits_time_group"])
            if int(main_table.num_rows) != int(num_rows):
                raise RuntimeError(
                    f"Row count mismatch for {source_parquet_path}: table={int(main_table.num_rows)} expected={int(num_rows)}"
                )
            hits_group_col = main_table.column("hits_time_group").combine_chunks()
            all_group_ids_by_row: list[list[int]] = []
            for row_idx in range(num_rows):
                scalar = hits_group_col[row_idx]
                if not scalar.is_valid:
                    all_group_ids_by_row.append([])
                    continue
                vals = np.asarray(scalar.as_py() or [], dtype=np.int64)
                if vals.size == 0:
                    all_group_ids_by_row.append([])
                    continue
                all_group_ids_by_row.append(np.unique(vals).astype(np.int64, copy=False).tolist())

            valid = (event_ids_np >= 0) & (event_ids_np < num_rows) & (group_ids_np >= 0)
            event_ids_v = event_ids_np[valid]
            group_ids_v = group_ids_np[valid]
            preds_v = preds_np[valid]

            pred_map: dict[tuple[int, int], np.ndarray] = {}
            for idx in range(int(event_ids_v.size)):
                key = (int(event_ids_v[idx]), int(group_ids_v[idx]))
                if key not in pred_map:
                    pred_map[key] = preds_v[idx]

            if pred_dim == 9:
                def _reshape_quant(arr: np.ndarray) -> np.ndarray:
                    return arr.reshape(3, 3)
            else:
                def _reshape_quant(arr: np.ndarray) -> np.ndarray:
                    return np.repeat(arr.reshape(3, 1), repeats=3, axis=1)

            out_cols: dict[str, pa.Array] = {
                "event_id": pa.array(np.arange(num_rows, dtype=np.int64)),
                "time_group_ids": pa.array(all_group_ids_by_row, type=pa.list_(pa.int64())),
            }
            for axis_idx, axis_name in enumerate(cls.TARGET_AXES):
                base_name = f"pred_pion_stop_{axis_name}"
                median_lists: list[list[float | None]] = []
                q16_lists: list[list[float | None]] = []
                q50_lists: list[list[float | None]] = []
                q84_lists: list[list[float | None]] = []
                for row_idx, tg_ids in enumerate(all_group_ids_by_row):
                    med_row: list[float | None] = []
                    q16_row: list[float | None] = []
                    q50_row: list[float | None] = []
                    q84_row: list[float | None] = []
                    for gid in tg_ids:
                        pred = pred_map.get((int(row_idx), int(gid)))
                        if pred is None:
                            med_row.append(None)
                            q16_row.append(None)
                            q50_row.append(None)
                            q84_row.append(None)
                            continue
                        vals = _reshape_quant(pred)[axis_idx]
                        med_row.append(float(vals[1]))
                        q16_row.append(float(vals[0]))
                        q50_row.append(float(vals[1]))
                        q84_row.append(float(vals[2]))
                    median_lists.append(med_row)
                    q16_lists.append(q16_row)
                    q50_lists.append(q50_row)
                    q84_lists.append(q84_row)
                out_cols[base_name] = pa.array(median_lists, type=pa.list_(pa.float32()))
                out_cols[f"{base_name}_q16"] = pa.array(q16_lists, type=pa.list_(pa.float32()))
                out_cols[f"{base_name}_q50"] = pa.array(q50_lists, type=pa.list_(pa.float32()))
                out_cols[f"{base_name}_q84"] = pa.array(q84_lists, type=pa.list_(pa.float32()))
            return pa.table(out_cols)

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

        if pred_dim == 9:
            preds_quant = preds_sorted.reshape(-1, 3, 3)
        else:
            preds_quant = np.repeat(preds_sorted.reshape(-1, 3, 1), repeats=3, axis=2)

        out_cols: dict[str, pa.Array] = {
            "event_id": pa.array(np.arange(num_rows, dtype=np.int64)),
            "time_group_ids": pa.ListArray.from_arrays(offsets, pa.array(group_sorted, type=pa.int64())),
        }
        for axis_idx, axis_name in enumerate(cls.TARGET_AXES):
            vals = preds_quant[:, axis_idx, :].astype(np.float32, copy=False)
            base_name = f"pred_pion_stop_{axis_name}"
            out_cols[base_name] = pa.ListArray.from_arrays(offsets, pa.array(vals[:, 1], type=pa.float32()))
            for q_idx, q_suffix in enumerate(cls.QUANTILE_SUFFIXES):
                out_cols[f"{base_name}_{q_suffix}"] = pa.ListArray.from_arrays(
                    offsets,
                    pa.array(vals[:, q_idx], type=pa.float32()),
                )
        return pa.table(out_cols)
