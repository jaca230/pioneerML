from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ....input_source import PredictionSet, TimeGroupPredictionSet
from ..graph_data_writer import GraphDataWriter


class TimeGroupGraphDataWriter(GraphDataWriter):
    """Shared helpers for writers that stitch graph/time-group predictions back to event rows."""

    def _resolve_source_event_ids(self, *, src_path: Path, num_rows: int) -> np.ndarray:
        cache = getattr(self, "_source_event_ids_cache_by_path", None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(self, "_source_event_ids_cache_by_path", cache)

        src_key = str(Path(src_path).expanduser().resolve())
        cached = cache.get(src_key)
        if isinstance(cached, np.ndarray):
            if int(cached.shape[0]) != int(num_rows):
                raise RuntimeError(
                    f"Cached event_id length mismatch for '{src_key}': {cached.shape[0]} vs expected {num_rows}."
                )
            return cached

        resolved = Path(src_key)
        if resolved.suffix.lower() == ".parquet":
            try:
                table = pq.read_table(resolved, columns=["event_id"])
                arr = table.column("event_id").combine_chunks().to_numpy(zero_copy_only=False)
                out = np.asarray(arr, dtype=np.int64)
            except Exception:
                out = np.arange(int(num_rows), dtype=np.int64)
        else:
            out = np.arange(int(num_rows), dtype=np.int64)

        if int(out.shape[0]) != int(num_rows):
            raise RuntimeError(
                f"Source event_id length mismatch for '{src_key}': {out.shape[0]} vs expected {num_rows}."
            )
        cache[src_key] = out
        return out

    @staticmethod
    def _row_offsets_for_event_span(
        *,
        event_ids_np: np.ndarray,
        start_event_id: int,
        stop_event_id: int,
    ) -> np.ndarray:
        row_count = int(stop_event_id) - int(start_event_id)
        if row_count < 0:
            raise ValueError(f"Invalid event span [{start_event_id}, {stop_event_id}).")
        offsets = np.zeros((row_count + 1,), dtype=np.int64)
        if row_count == 0 or event_ids_np.size == 0:
            return offsets

        local_ids = np.asarray(event_ids_np, dtype=np.int64) - int(start_event_id)
        if np.any(local_ids < 0) or np.any(local_ids >= int(row_count)):
            raise ValueError("Event ids are outside the requested event span.")
        counts = np.bincount(local_ids, minlength=row_count).astype(np.int64, copy=False)
        offsets[1:] = np.cumsum(counts, dtype=np.int64)
        return offsets

    @staticmethod
    def normalize_time_group_inputs(
        *,
        time_group_event_ids_np: np.ndarray | None,
        time_group_ids_np: np.ndarray | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if (time_group_event_ids_np is None) != (time_group_ids_np is None):
            raise ValueError(
                "time_group_event_ids_np and time_group_ids_np must either both be provided or both be None."
            )
        if time_group_event_ids_np is None:
            return None, None
        return (
            np.asarray(time_group_event_ids_np, dtype=np.int64),
            np.asarray(time_group_ids_np, dtype=np.int64),
        )

    @staticmethod
    def stitch_predictions_to_events(
        *,
        prediction_event_ids_np: np.ndarray,
        prediction_columns: Mapping[str, np.ndarray],
        num_rows: int,
        source_event_ids_np: np.ndarray | None = None,
        value_types: Mapping[str, pa.DataType] | None = None,
        time_group_event_ids_np: np.ndarray | None = None,
        time_group_ids_np: np.ndarray | None = None,
        group_id_column: str = "time_group_ids",
    ) -> pa.Table:
        event_ids_arr = np.asarray(prediction_event_ids_np, dtype=np.int64)
        resolved_rows = TimeGroupGraphDataWriter.normalize_num_rows(event_ids_arr, int(num_rows))
        valid_mask = TimeGroupGraphDataWriter.validated_event_indices(event_ids_np=event_ids_arr, num_rows=resolved_rows)
        event_ids_valid = event_ids_arr[valid_mask]

        # Predictions are expected to be row-local aligned (stable event order from loaders/writers).
        if event_ids_valid.size > 1 and np.any(event_ids_valid[1:] < event_ids_valid[:-1]):
            raise ValueError(
                "prediction_event_ids_np must be non-decreasing (row-local aligned per event). "
                "Disable shuffling for inference/export writers."
            )
        counts = np.bincount(event_ids_valid, minlength=resolved_rows).astype(np.int64, copy=False)
        offsets = np.zeros((resolved_rows + 1,), dtype=np.int64)
        offsets[1:] = np.cumsum(counts, dtype=np.int64)

        if source_event_ids_np is None:
            source_event_ids = np.arange(int(resolved_rows), dtype=np.int64)
        else:
            source_event_ids = np.asarray(source_event_ids_np, dtype=np.int64)
            if int(source_event_ids.shape[0]) != int(resolved_rows):
                raise ValueError(
                    f"source_event_ids_np length mismatch: {source_event_ids.shape[0]} vs expected {resolved_rows}."
                )

        arrays: dict[str, pa.Array] = {"event_id": pa.array(source_event_ids, type=pa.int64())}
        type_map = dict(value_types or {})
        for col, raw in prediction_columns.items():
            values = np.asarray(raw)
            if values.shape[0] != event_ids_arr.shape[0]:
                raise ValueError(
                    f"Column '{col}' has leading dim {values.shape[0]} but expected {event_ids_arr.shape[0]}."
                )
            valid_vals = values[valid_mask]
            arrays[str(col)] = TimeGroupGraphDataWriter.list_array_from_prediction_chunk(
                offsets=offsets,
                values=valid_vals,
                value_type=type_map.get(str(col)),
            )

        if time_group_event_ids_np is not None and time_group_ids_np is not None:
            tg_event_ids = np.asarray(time_group_event_ids_np, dtype=np.int64)
            tg_ids = np.asarray(time_group_ids_np, dtype=np.int64)
            if tg_event_ids.shape[0] != tg_ids.shape[0]:
                raise ValueError(
                    f"time_group_event_ids and time_group_ids length mismatch: "
                    f"{tg_event_ids.shape[0]} vs {tg_ids.shape[0]}."
                )
            tg_valid = TimeGroupGraphDataWriter.validated_event_indices(event_ids_np=tg_event_ids, num_rows=resolved_rows)
            tg_event_ids = tg_event_ids[tg_valid]
            tg_ids = tg_ids[tg_valid]
            if tg_event_ids.size > 0:
                if tg_event_ids.size > 1 and np.any(tg_event_ids[1:] < tg_event_ids[:-1]):
                    raise ValueError(
                        "time_group_event_ids_np must be non-decreasing (row-local aligned per event). "
                        "Disable shuffling for inference/export writers."
                    )
                tg_counts = np.bincount(tg_event_ids, minlength=resolved_rows).astype(np.int64, copy=False)
            else:
                tg_counts = np.zeros((resolved_rows,), dtype=np.int64)
            tg_offsets = np.zeros((resolved_rows + 1,), dtype=np.int64)
            tg_offsets[1:] = np.cumsum(tg_counts, dtype=np.int64)
            arrays[str(group_id_column)] = TimeGroupGraphDataWriter.list_array_from_prediction_chunk(
                offsets=tg_offsets,
                values=tg_ids.astype(np.int64, copy=False),
                value_type=pa.int64(),
            )

        return pa.table(arrays)

    @staticmethod
    def stitch_predictions_to_event_span(
        *,
        prediction_event_ids_np: np.ndarray,
        prediction_columns: Mapping[str, np.ndarray],
        num_rows: int,
        start_event_id: int,
        stop_event_id: int | None = None,
        source_event_ids_np: np.ndarray | None = None,
        value_types: Mapping[str, pa.DataType] | None = None,
        time_group_event_ids_np: np.ndarray | None = None,
        time_group_ids_np: np.ndarray | None = None,
        group_id_column: str = "time_group_ids",
    ) -> tuple[pa.Table, int]:
        event_ids_arr = np.asarray(prediction_event_ids_np, dtype=np.int64)
        resolved_rows = TimeGroupGraphDataWriter.normalize_num_rows(event_ids_arr, int(num_rows))
        span_start = int(start_event_id)
        if span_start < 0 or span_start > int(resolved_rows):
            raise ValueError(
                f"start_event_id must be within [0, {resolved_rows}] (got {span_start})."
            )

        valid_mask = TimeGroupGraphDataWriter.validated_event_indices(event_ids_np=event_ids_arr, num_rows=resolved_rows)
        event_ids_valid = event_ids_arr[valid_mask]
        if event_ids_valid.size > 1 and np.any(event_ids_valid[1:] < event_ids_valid[:-1]):
            raise ValueError(
                "prediction_event_ids_np must be non-decreasing (row-local aligned per event). "
                "Disable shuffling for inference/export writers."
            )
        if event_ids_valid.size > 0 and int(event_ids_valid[0]) < int(span_start):
            raise ValueError(
                "prediction_event_ids_np is out of order for streaming writes; "
                "encountered an event id earlier than the current write cursor."
            )

        inferred_stop = int(event_ids_valid[-1]) + 1 if event_ids_valid.size > 0 else int(span_start)
        span_stop = int(inferred_stop if stop_event_id is None else int(stop_event_id))
        if span_stop < int(span_start) or span_stop > int(resolved_rows):
            raise ValueError(
                f"stop_event_id must be within [{span_start}, {resolved_rows}] (got {span_stop})."
            )
        if event_ids_valid.size > 0 and int(event_ids_valid[-1]) >= int(span_stop):
            raise ValueError("prediction_event_ids_np contains ids outside the requested event span.")

        offsets = TimeGroupGraphDataWriter._row_offsets_for_event_span(
            event_ids_np=event_ids_valid,
            start_event_id=span_start,
            stop_event_id=span_stop,
        )

        if source_event_ids_np is None:
            source_event_ids = np.arange(int(resolved_rows), dtype=np.int64)
        else:
            source_event_ids = np.asarray(source_event_ids_np, dtype=np.int64)
            if int(source_event_ids.shape[0]) != int(resolved_rows):
                raise ValueError(
                    f"source_event_ids_np length mismatch: {source_event_ids.shape[0]} vs expected {resolved_rows}."
                )

        arrays: dict[str, pa.Array] = {
            "event_id": pa.array(source_event_ids[int(span_start) : int(span_stop)], type=pa.int64())
        }
        type_map = dict(value_types or {})
        for col, raw in prediction_columns.items():
            values = np.asarray(raw)
            if values.shape[0] != event_ids_arr.shape[0]:
                raise ValueError(
                    f"Column '{col}' has leading dim {values.shape[0]} but expected {event_ids_arr.shape[0]}."
                )
            valid_vals = values[valid_mask]
            arrays[str(col)] = TimeGroupGraphDataWriter.list_array_from_prediction_chunk(
                offsets=offsets,
                values=valid_vals,
                value_type=type_map.get(str(col)),
            )

        if time_group_event_ids_np is not None and time_group_ids_np is not None:
            tg_event_ids = np.asarray(time_group_event_ids_np, dtype=np.int64)
            tg_ids = np.asarray(time_group_ids_np, dtype=np.int64)
            if tg_event_ids.shape[0] != tg_ids.shape[0]:
                raise ValueError(
                    f"time_group_event_ids and time_group_ids length mismatch: "
                    f"{tg_event_ids.shape[0]} vs {tg_ids.shape[0]}."
                )
            tg_valid = TimeGroupGraphDataWriter.validated_event_indices(event_ids_np=tg_event_ids, num_rows=resolved_rows)
            tg_event_ids = tg_event_ids[tg_valid]
            tg_ids = tg_ids[tg_valid]
            if tg_event_ids.size > 1 and np.any(tg_event_ids[1:] < tg_event_ids[:-1]):
                raise ValueError(
                    "time_group_event_ids_np must be non-decreasing (row-local aligned per event). "
                    "Disable shuffling for inference/export writers."
                )
            span_mask = (tg_event_ids >= int(span_start)) & (tg_event_ids < int(span_stop))
            tg_event_ids = tg_event_ids[span_mask]
            tg_ids = tg_ids[span_mask]
            tg_offsets = TimeGroupGraphDataWriter._row_offsets_for_event_span(
                event_ids_np=tg_event_ids,
                start_event_id=span_start,
                stop_event_id=span_stop,
            )
            arrays[str(group_id_column)] = TimeGroupGraphDataWriter.list_array_from_prediction_chunk(
                offsets=tg_offsets,
                values=tg_ids.astype(np.int64, copy=False),
                value_type=pa.int64(),
            )

        return pa.table(arrays), int(span_stop)

    def chunk_state(
        self,
        *,
        prediction_set: PredictionSet,
        output_dir: Path,
        output_path: str | None,
        write_timestamped: bool,
        timestamp: str,
    ) -> dict[str, object]:
        prediction_set.validate()
        schema = self.output_schema()
        value_types = schema.value_types()
        state = super().chunk_state(
            src_path=prediction_set.src_path,
            prediction_event_ids_np=prediction_set.prediction_event_ids_np,
            prediction_columns=schema.prediction_columns(model_outputs_by_name=prediction_set.model_outputs_by_name),
            num_rows=prediction_set.num_rows,
            output_dir=output_dir,
            output_path=output_path,
            write_timestamped=write_timestamped,
            timestamp=timestamp,
            value_types=(value_types if value_types else None),
        )
        tg_event_ids, tg_ids = self.normalize_time_group_inputs(
            time_group_event_ids_np=(
                prediction_set.time_group_event_ids_np if isinstance(prediction_set, TimeGroupPredictionSet) else None
            ),
            time_group_ids_np=(
                prediction_set.time_group_ids_np if isinstance(prediction_set, TimeGroupPredictionSet) else None
            ),
        )
        if tg_event_ids is not None:
            state["time_group_event_ids_np"] = tg_event_ids
            state["time_group_ids_np"] = tg_ids
            state["group_id_column"] = (
                prediction_set.group_id_column if isinstance(prediction_set, TimeGroupPredictionSet) else "time_group_ids"
            )
        return state
