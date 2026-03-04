from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

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
        if not bool(state.get("streaming", False)):
            return
        prediction_columns = state.get(self.prediction_columns_key, state[self.fallback_prediction_columns_key])
        time_group_event_ids = (
            None if self.time_group_event_ids_key is None else state.get(self.time_group_event_ids_key)
        )
        time_group_ids = None if self.time_group_ids_key is None else state.get(self.time_group_ids_key)
        state["table"] = owner.stitch_predictions_to_events(
            prediction_event_ids_np=state[self.prediction_event_ids_key],
            prediction_columns=prediction_columns,
            num_rows=int(state[self.num_rows_key]),
            value_types=state.get(self.value_types_key),
            time_group_event_ids_np=time_group_event_ids,
            time_group_ids_np=time_group_ids,
            group_id_column=str(state.get(self.group_id_column_key, self.default_group_id_column)),
        )
