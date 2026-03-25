from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import numpy as np

from .base_time_group_writer_stage import BaseTimeGroupWriterStage


class FinalizeBufferedWritesStage(BaseTimeGroupWriterStage):
    name = "finalize_buffered_writes"

    @staticmethod
    def _empty_prediction_columns(column_names: list[str]) -> dict[str, np.ndarray]:
        return {name: np.empty((0,), dtype=np.float32) for name in column_names}

    def run_time_group_writer(self, *, state: MutableMapping[str, Any], owner) -> None:
        if bool(state.get("streaming", False)):
            source_contexts = list(state.get("source_contexts") or [])
            if not source_contexts:
                return
            sink_entries = dict(state.get("open_stream_sinks") or {})
            stream_next = state.get("stream_next_event_by_src")
            if not isinstance(stream_next, dict):
                stream_next = {}
            stream_buffers = state.get("stream_buffers_by_src")
            if not isinstance(stream_buffers, dict):
                stream_buffers = {}
            value_types = state.get("value_types")
            if value_types is None:
                resolved_value_types = owner.output_schema().value_types()
                value_types = (resolved_value_types if resolved_value_types else None)

            template_columns = state.get("prediction_columns")
            empty_prediction_columns: dict[str, np.ndarray]
            if isinstance(template_columns, dict) and template_columns:
                empty_prediction_columns = {}
                for key, arr in template_columns.items():
                    values = np.asarray(arr)
                    tail_shape = tuple(int(v) for v in values.shape[1:])
                    empty_prediction_columns[str(key)] = np.empty((0, *tail_shape), dtype=values.dtype)
            else:
                empty_prediction_columns = self._empty_prediction_columns(
                    column_names=list(owner.output_schema().column_names())
                )

            empty_event_ids = np.empty((0,), dtype=np.int64)
            group_id_column = str(state.get("group_id_column", "time_group_ids"))
            for ctx in source_contexts:
                src_path = Path(ctx["src_path"]).expanduser().resolve()
                src_key = str(src_path)
                entry = sink_entries.get(src_key)
                if entry is None:
                    raise RuntimeError(f"Missing streaming sink entry for source '{src_path}'.")
                start_event = int(stream_next.get(src_key, 0))
                num_rows = int(ctx["num_rows"])
                source_event_ids = owner._resolve_source_event_ids(src_path=src_path, num_rows=num_rows)
                if start_event < 0 or start_event > num_rows:
                    raise RuntimeError(
                        f"Invalid streaming cursor for source '{src_path}': {start_event} not in [0, {num_rows}]."
                    )

                carry = stream_buffers.get(src_key)
                if isinstance(carry, dict):
                    carry_event_ids = np.asarray(carry.get("prediction_event_ids_np"), dtype=np.int64)
                    carry_prediction_columns = {
                        str(k): np.asarray(v) for k, v in dict(carry.get("prediction_columns") or {}).items()
                    }
                    carry_tg_event_ids_raw = carry.get("time_group_event_ids_np")
                    carry_tg_ids_raw = carry.get("time_group_ids_np")
                    if carry_tg_event_ids_raw is None or carry_tg_ids_raw is None:
                        carry_tg_event_ids = None
                        carry_tg_ids = None
                    else:
                        carry_tg_event_ids = np.asarray(carry_tg_event_ids_raw, dtype=np.int64)
                        carry_tg_ids = np.asarray(carry_tg_ids_raw, dtype=np.int64)
                else:
                    carry_event_ids = empty_event_ids
                    carry_prediction_columns = dict(empty_prediction_columns)
                    carry_tg_event_ids = empty_event_ids
                    carry_tg_ids = empty_event_ids

                if not carry_prediction_columns:
                    carry_prediction_columns = dict(empty_prediction_columns)

                tail_table, tail_stop = owner.stitch_predictions_to_event_span(
                    prediction_event_ids_np=carry_event_ids,
                    prediction_columns=carry_prediction_columns,
                    num_rows=num_rows,
                    start_event_id=start_event,
                    stop_event_id=num_rows,
                    source_event_ids_np=source_event_ids,
                    value_types=value_types,
                    time_group_event_ids_np=carry_tg_event_ids,
                    time_group_ids_np=carry_tg_ids,
                    group_id_column=group_id_column,
                )
                owner.output_backend.append_chunk(sink=entry["sink"], table=tail_table)
                stream_next[src_key] = int(tail_stop)
                stream_buffers[src_key] = {
                    "prediction_event_ids_np": empty_event_ids,
                    "prediction_columns": {
                        str(k): np.empty((0, *np.asarray(v).shape[1:]), dtype=np.asarray(v).dtype)
                        for k, v in carry_prediction_columns.items()
                    },
                    "time_group_event_ids_np": empty_event_ids if carry_tg_event_ids is not None else None,
                    "time_group_ids_np": empty_event_ids if carry_tg_ids is not None else None,
                }
            state["stream_next_event_by_src"] = stream_next
            state["stream_buffers_by_src"] = stream_buffers
            return

        source_contexts = list(state.get("source_contexts") or [])
        buffered = list(state.get("buffered_chunks") or [])
        output_dir = Path(state["output_dir"])
        output_path = state.get("output_path")
        write_timestamped = bool(state.get("write_timestamped", False))
        timestamp = str(state.get("timestamp", owner.timestamp()))
        column_names = [str(c) for c in list(state.get("prediction_column_names") or [])]

        written = state.get("written_prediction_paths")
        if not isinstance(written, list):
            written = []
            state["written_prediction_paths"] = written
        written_ts = state.get("written_timestamped_paths")
        if not isinstance(written_ts, list):
            written_ts = []
            state["written_timestamped_paths"] = written_ts

        for idx, ctx in enumerate(source_contexts):
            src_path = Path(ctx["src_path"]).expanduser().resolve()
            num_rows = int(ctx["num_rows"])
            source_event_ids = owner._resolve_source_event_ids(src_path=src_path, num_rows=num_rows)
            chunks = [c for c in buffered if Path(c["src_path"]).expanduser().resolve() == src_path]
            if chunks:
                event_ids = np.concatenate([c["prediction_event_ids_np"] for c in chunks], axis=0)
                pred_cols: dict[str, np.ndarray] = {}
                for key in chunks[0]["prediction_columns"].keys():
                    pred_cols[str(key)] = np.concatenate([np.asarray(c["prediction_columns"][key]) for c in chunks], axis=0)
                tg_event = (
                    None
                    if chunks[0].get("time_group_event_ids_np") is None
                    else np.concatenate([c["time_group_event_ids_np"] for c in chunks], axis=0)
                )
                tg_ids = (
                    None
                    if chunks[0].get("time_group_ids_np") is None
                    else np.concatenate([c["time_group_ids_np"] for c in chunks], axis=0)
                )
                value_types = chunks[0].get("value_types")
                group_id_column = str(chunks[0].get("group_id_column", "time_group_ids"))
            else:
                event_ids = np.empty((0,), dtype=np.int64)
                pred_cols = self._empty_prediction_columns(column_names=column_names)
                tg_event = None
                tg_ids = None
                value_types = state.get("value_types")
                group_id_column = str(state.get("group_id_column", "time_group_ids"))

            table = owner.stitch_predictions_to_events(
                prediction_event_ids_np=event_ids,
                prediction_columns=pred_cols,
                num_rows=num_rows,
                source_event_ids_np=source_event_ids,
                value_types=value_types,
                time_group_event_ids_np=tg_event,
                time_group_ids_np=tg_ids,
                group_id_column=group_id_column,
            )

            scoped_output_path = str(output_path) if (output_path and len(source_contexts) == 1 and idx == 0) else None
            pred_path = owner.resolve_prediction_output_path(
                src_path=src_path,
                output_dir=output_dir,
                output_path=scoped_output_path,
            )
            pred, ts = owner.write_table_with_optional_timestamp(
                table=table,
                pred_path=pred_path,
                output_dir=output_dir,
                src_path=src_path,
                write_timestamped=write_timestamped,
                timestamp=timestamp,
            )
            written.append(pred)
            if ts is not None:
                written_ts.append(ts)
