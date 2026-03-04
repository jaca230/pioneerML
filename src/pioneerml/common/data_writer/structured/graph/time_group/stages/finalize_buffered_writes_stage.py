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
