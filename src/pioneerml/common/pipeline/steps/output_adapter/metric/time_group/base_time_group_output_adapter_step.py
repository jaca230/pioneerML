from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..base_metric_output_adapter_step import BaseMetricOutputAdapterStep


class BaseTimeGroupOutputAdapterStep(BaseMetricOutputAdapterStep):
    """Output-adapter helpers for stitching time-group predictions back to event rows."""

    @staticmethod
    def _normalize_num_rows(event_ids_np: np.ndarray, num_rows: int) -> int:
        if int(num_rows) > 0:
            return int(num_rows)
        return (int(event_ids_np.max()) + 1) if event_ids_np.size > 0 else 0

    @staticmethod
    def _validated_event_indices(*, event_ids_np: np.ndarray, num_rows: int) -> np.ndarray:
        return (event_ids_np >= 0) & (event_ids_np < int(num_rows))

    @staticmethod
    def _stable_group_order(event_ids_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if event_ids_np.size == 0:
            empty = np.zeros((0,), dtype=np.int64)
            return empty, empty, np.zeros((1,), dtype=np.int64)
        order = np.argsort(event_ids_np, kind="stable")
        event_sorted = event_ids_np[order]
        counts = np.bincount(event_sorted).astype(np.int64, copy=False)
        offsets = np.zeros((counts.shape[0] + 1,), dtype=np.int64)
        offsets[1:] = np.cumsum(counts, dtype=np.int64)
        return order, event_sorted, offsets

    @staticmethod
    def _list_array_from_group_values(*, offsets: np.ndarray, values: np.ndarray, value_type: pa.DataType | None = None) -> pa.Array:
        if values.ndim == 1:
            arr = pa.array(values, type=value_type)
            return pa.ListArray.from_arrays(offsets, arr)

        if values.ndim != 2:
            raise ValueError(f"Prediction values must be 1D or 2D, got shape {tuple(values.shape)}.")

        inner_size = int(values.shape[1])
        flat = values.reshape(-1)
        inner_values = pa.array(flat, type=value_type)
        inner = pa.FixedSizeListArray.from_arrays(inner_values, list_size=inner_size)
        return pa.ListArray.from_arrays(offsets, inner)

    @classmethod
    def stitch_time_group_predictions_to_events(
        cls,
        *,
        event_ids_np: np.ndarray,
        prediction_columns: Mapping[str, np.ndarray],
        num_rows: int,
        value_types: Mapping[str, pa.DataType] | None = None,
    ) -> pa.Table:
        """Build event-level list columns from per-time-group predictions.

        - `event_ids_np`: shape [num_groups], each entry is the event row index.
        - `prediction_columns`: column_name -> array with first dimension `num_groups`.
        - returns one row per event with list-valued prediction columns.
        """
        event_ids_arr = np.asarray(event_ids_np, dtype=np.int64)
        resolved_rows = cls._normalize_num_rows(event_ids_arr, int(num_rows))
        valid_mask = cls._validated_event_indices(event_ids_np=event_ids_arr, num_rows=resolved_rows)
        event_ids_valid = event_ids_arr[valid_mask]

        order, event_sorted, offsets_short = cls._stable_group_order(event_ids_valid)
        counts = np.bincount(event_sorted, minlength=resolved_rows).astype(np.int64, copy=False)
        offsets = np.zeros((resolved_rows + 1,), dtype=np.int64)
        offsets[1:] = np.cumsum(counts, dtype=np.int64)

        arrays: dict[str, pa.Array] = {}
        type_map = dict(value_types or {})

        for col, raw in prediction_columns.items():
            values = np.asarray(raw)
            if values.shape[0] != event_ids_arr.shape[0]:
                raise ValueError(
                    f"Column '{col}' has leading dim {values.shape[0]} but expected {event_ids_arr.shape[0]}."
                )
            valid_vals = values[valid_mask]
            sorted_vals = valid_vals[order] if valid_vals.shape[0] > 0 else valid_vals
            arrays[str(col)] = cls._list_array_from_group_values(
                offsets=offsets,
                values=sorted_vals,
                value_type=type_map.get(str(col)),
            )
        return pa.table(arrays)

    def write_non_streamed_time_group_predictions(
        self,
        *,
        output_dir: Path,
        output_path: str | None,
        write_timestamped: bool,
        timestamp: str,
        validated_files: list[str],
        event_ids_np: np.ndarray,
        prediction_columns: Mapping[str, np.ndarray],
        value_types: Mapping[str, pa.DataType] | None,
        num_rows: int,
    ) -> tuple[list[str], list[str]]:
        """Write non-streamed time-group predictions with optional per-file splitting."""
        per_file_output_paths: list[str] = []
        per_file_timestamped_paths: list[str] = []

        event_ids_arr = np.asarray(event_ids_np, dtype=np.int64)
        columns_arr = {k: np.asarray(v) for k, v in prediction_columns.items()}
        for name, arr in columns_arr.items():
            if arr.shape[0] != event_ids_arr.shape[0]:
                raise ValueError(
                    f"Column '{name}' has leading dim {arr.shape[0]} but expected {event_ids_arr.shape[0]}."
                )

        if output_path and len(validated_files) != 1:
            raise ValueError("output_path is only supported when exactly one input parquet file is provided.")

        if validated_files:
            row_counts = [int(pq.ParquetFile(p).metadata.num_rows) for p in validated_files]
            start = 0
            for src_file, n_rows in zip(validated_files, row_counts, strict=True):
                end = start + n_rows
                mask = (event_ids_arr >= start) & (event_ids_arr < end)
                local_event_ids = event_ids_arr[mask] - start
                local_cols = {k: arr[mask] for k, arr in columns_arr.items()}

                table = self.stitch_time_group_predictions_to_events(
                    event_ids_np=local_event_ids,
                    prediction_columns=local_cols,
                    num_rows=n_rows,
                    value_types=value_types,
                )
                pred_path = (
                    Path(output_path)
                    if (output_path and len(validated_files) == 1)
                    else output_dir / f"{Path(src_file).stem}_preds.parquet"
                )
                self.atomic_write_table(table=table, dst_path=pred_path)
                per_file_output_paths.append(str(pred_path))

                if write_timestamped:
                    ts_path = output_dir / f"{Path(src_file).stem}_preds_{timestamp}.parquet"
                    self.atomic_write_table(table=table, dst_path=ts_path)
                    per_file_timestamped_paths.append(str(ts_path))
                start = end
            return per_file_output_paths, per_file_timestamped_paths

        table = self.stitch_time_group_predictions_to_events(
            event_ids_np=event_ids_arr,
            prediction_columns=columns_arr,
            num_rows=int(num_rows),
            value_types=value_types,
        )
        pred_path = Path(output_path) if output_path else (output_dir / "preds.parquet")
        self.atomic_write_table(table=table, dst_path=pred_path)
        per_file_output_paths.append(str(pred_path))
        if write_timestamped:
            ts_path = output_dir / f"preds_{timestamp}.parquet"
            self.atomic_write_table(table=table, dst_path=ts_path)
            per_file_timestamped_paths.append(str(ts_path))
        return per_file_output_paths, per_file_timestamped_paths
