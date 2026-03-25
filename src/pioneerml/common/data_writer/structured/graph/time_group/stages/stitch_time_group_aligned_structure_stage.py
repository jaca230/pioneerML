from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import numpy as np

from .base_time_group_writer_stage import BaseTimeGroupWriterStage


class StitchTimeGroupAlignedStructureStage(BaseTimeGroupWriterStage):
    """Stitch predictions back to event rows with optional time-group-id stitching."""

    name = "stitch_structure"

    def __init__(
        self,
        *,
        prediction_event_ids_key: str,
        time_group_event_ids_key: str | None = None,
        time_group_ids_key: str | None = None,
        prediction_columns_key: str = "output_columns",
        fallback_prediction_columns_key: str = "prediction_columns",
        num_rows_key: str = "num_rows",
        value_types_key: str = "value_types",
        group_id_column_key: str = "group_id_column",
        default_group_id_column: str = "time_group_ids",
    ) -> None:
        self.prediction_event_ids_key = str(prediction_event_ids_key)
        self.time_group_event_ids_key = None if time_group_event_ids_key is None else str(time_group_event_ids_key)
        self.time_group_ids_key = None if time_group_ids_key is None else str(time_group_ids_key)
        self.prediction_columns_key = str(prediction_columns_key)
        self.fallback_prediction_columns_key = str(fallback_prediction_columns_key)
        self.num_rows_key = str(num_rows_key)
        self.value_types_key = str(value_types_key)
        self.group_id_column_key = str(group_id_column_key)
        self.default_group_id_column = str(default_group_id_column)

    def run_time_group_writer(self, *, state: MutableMapping[str, Any], owner) -> None:
        prediction_columns = state.get(self.prediction_columns_key, state[self.fallback_prediction_columns_key])
        time_group_event_ids = (
            None if self.time_group_event_ids_key is None else state.get(self.time_group_event_ids_key)
        )
        time_group_ids = None if self.time_group_ids_key is None else state.get(self.time_group_ids_key)
        group_id_column = str(state.get(self.group_id_column_key, self.default_group_id_column))
        src_path = Path(state["src_path"]).expanduser().resolve()
        source_event_ids = owner._resolve_source_event_ids(
            src_path=src_path,
            num_rows=int(state[self.num_rows_key]),
        )
        if not bool(state.get("streaming", False)):
            state["table"] = owner.stitch_predictions_to_events(
                prediction_event_ids_np=state[self.prediction_event_ids_key],
                prediction_columns=prediction_columns,
                num_rows=int(state[self.num_rows_key]),
                source_event_ids_np=source_event_ids,
                value_types=state.get(self.value_types_key),
                time_group_event_ids_np=time_group_event_ids,
                time_group_ids_np=time_group_ids,
                group_id_column=group_id_column,
            )
            return

        src_key = str(src_path)
        stream_next = state.get("stream_next_event_by_src")
        if not isinstance(stream_next, dict):
            stream_next = {}
        stream_buffers = state.get("stream_buffers_by_src")
        if not isinstance(stream_buffers, dict):
            stream_buffers = {}

        span_start = int(stream_next.get(src_key, 0))
        chunk_event_ids = np.asarray(state[self.prediction_event_ids_key], dtype=np.int64)
        chunk_prediction_columns = {str(k): np.asarray(v) for k, v in dict(prediction_columns).items()}
        chunk_tg_event_ids = (
            None if time_group_event_ids is None else np.asarray(time_group_event_ids, dtype=np.int64)
        )
        chunk_tg_ids = None if time_group_ids is None else np.asarray(time_group_ids, dtype=np.int64)

        prev = stream_buffers.get(src_key)
        if isinstance(prev, dict):
            prev_event_ids = np.asarray(prev.get("prediction_event_ids_np"), dtype=np.int64)
            prev_prediction_columns = {str(k): np.asarray(v) for k, v in dict(prev.get("prediction_columns") or {}).items()}
            if prev_prediction_columns:
                merged_prediction_columns: dict[str, np.ndarray] = {}
                for key, chunk_vals in chunk_prediction_columns.items():
                    prev_vals = prev_prediction_columns.get(key)
                    if prev_vals is None:
                        prev_vals = np.empty((0, *np.asarray(chunk_vals).shape[1:]), dtype=np.asarray(chunk_vals).dtype)
                    prev_vals = np.asarray(prev_vals)
                    merged_prediction_columns[key] = np.concatenate([prev_vals, chunk_vals], axis=0)
            else:
                merged_prediction_columns = dict(chunk_prediction_columns)
            merged_event_ids = np.concatenate([prev_event_ids, chunk_event_ids], axis=0)
            prev_tg_event_ids = prev.get("time_group_event_ids_np")
            prev_tg_ids = prev.get("time_group_ids_np")
            if prev_tg_event_ids is None or prev_tg_ids is None or chunk_tg_event_ids is None or chunk_tg_ids is None:
                merged_tg_event_ids = chunk_tg_event_ids
                merged_tg_ids = chunk_tg_ids
            else:
                merged_tg_event_ids = np.concatenate([np.asarray(prev_tg_event_ids, dtype=np.int64), chunk_tg_event_ids], axis=0)
                merged_tg_ids = np.concatenate([np.asarray(prev_tg_ids, dtype=np.int64), chunk_tg_ids], axis=0)
        else:
            merged_event_ids = chunk_event_ids
            merged_prediction_columns = dict(chunk_prediction_columns)
            merged_tg_event_ids = chunk_tg_event_ids
            merged_tg_ids = chunk_tg_ids

        if merged_event_ids.size > 1 and np.any(merged_event_ids[1:] < merged_event_ids[:-1]):
            raise ValueError(
                "prediction_event_ids_np must be non-decreasing for streaming writes. "
                "Disable shuffling for inference/export writers."
            )

        if merged_event_ids.size == 0:
            flush_event_ids = np.empty((0,), dtype=np.int64)
            carry_event_ids = np.empty((0,), dtype=np.int64)
            flush_stop = int(span_start)
        else:
            max_event = int(merged_event_ids[-1])
            flush_stop = int(max(int(span_start), max_event))
            flush_mask = merged_event_ids < int(max_event)
            carry_mask = ~flush_mask
            flush_event_ids = merged_event_ids[flush_mask]
            carry_event_ids = merged_event_ids[carry_mask]

        flush_prediction_columns: dict[str, np.ndarray] = {}
        carry_prediction_columns: dict[str, np.ndarray] = {}
        for key, values in merged_prediction_columns.items():
            arr = np.asarray(values)
            if merged_event_ids.size == 0:
                flush_prediction_columns[str(key)] = np.empty((0, *arr.shape[1:]), dtype=arr.dtype)
                carry_prediction_columns[str(key)] = np.empty((0, *arr.shape[1:]), dtype=arr.dtype)
            else:
                flush_prediction_columns[str(key)] = arr[flush_mask]
                carry_prediction_columns[str(key)] = arr[carry_mask]

        if merged_tg_event_ids is not None and merged_tg_ids is not None and merged_event_ids.size > 0:
            flush_tg_mask = merged_tg_event_ids < int(flush_stop)
            carry_tg_mask = ~flush_tg_mask
            flush_tg_event_ids = merged_tg_event_ids[flush_tg_mask]
            flush_tg_ids = merged_tg_ids[flush_tg_mask]
            carry_tg_event_ids = merged_tg_event_ids[carry_tg_mask]
            carry_tg_ids = merged_tg_ids[carry_tg_mask]
        elif merged_tg_event_ids is not None and merged_tg_ids is not None:
            flush_tg_event_ids = np.empty((0,), dtype=np.int64)
            flush_tg_ids = np.empty((0,), dtype=np.int64)
            carry_tg_event_ids = np.empty((0,), dtype=np.int64)
            carry_tg_ids = np.empty((0,), dtype=np.int64)
        else:
            flush_tg_event_ids = None
            flush_tg_ids = None
            carry_tg_event_ids = None
            carry_tg_ids = None

        table, span_stop = owner.stitch_predictions_to_event_span(
            prediction_event_ids_np=flush_event_ids,
            prediction_columns=flush_prediction_columns,
            num_rows=int(state[self.num_rows_key]),
            start_event_id=span_start,
            stop_event_id=flush_stop,
            source_event_ids_np=source_event_ids,
            value_types=state.get(self.value_types_key),
            time_group_event_ids_np=flush_tg_event_ids,
            time_group_ids_np=flush_tg_ids,
            group_id_column=group_id_column,
        )
        stream_next[src_key] = int(span_stop)
        stream_buffers[src_key] = {
            "prediction_event_ids_np": carry_event_ids,
            "prediction_columns": carry_prediction_columns,
            "time_group_event_ids_np": carry_tg_event_ids,
            "time_group_ids_np": carry_tg_ids,
        }
        state["stream_next_event_by_src"] = stream_next
        state["stream_buffers_by_src"] = stream_buffers
        state["table"] = table
