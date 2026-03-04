from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pyarrow as pa

from ....input_source import PredictionSet, TimeGroupPredictionSet
from ..graph_data_writer import GraphDataWriter


class TimeGroupGraphDataWriter(GraphDataWriter):
    """Shared helpers for writers that stitch graph/time-group predictions back to event rows."""

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
        value_types: Mapping[str, pa.DataType] | None = None,
        time_group_event_ids_np: np.ndarray | None = None,
        time_group_ids_np: np.ndarray | None = None,
        group_id_column: str = "time_group_ids",
    ) -> pa.Table:
        event_ids_arr = np.asarray(prediction_event_ids_np, dtype=np.int64)
        resolved_rows = TimeGroupGraphDataWriter.normalize_num_rows(event_ids_arr, int(num_rows))
        valid_mask = TimeGroupGraphDataWriter.validated_event_indices(event_ids_np=event_ids_arr, num_rows=resolved_rows)
        event_ids_valid = event_ids_arr[valid_mask]

        order, event_sorted = TimeGroupGraphDataWriter.stable_group_order(event_ids_valid)
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
            arrays[str(col)] = TimeGroupGraphDataWriter.list_array_from_prediction_chunk(
                offsets=offsets,
                values=sorted_vals,
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
                tg_order = np.lexsort((tg_ids, tg_event_ids))
                tg_event_ids = tg_event_ids[tg_order]
                tg_ids = tg_ids[tg_order]
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
