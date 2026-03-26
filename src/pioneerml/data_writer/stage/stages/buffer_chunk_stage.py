from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import numpy as np

from .base_stage import BaseWriterStage


class BufferChunkStage(BaseWriterStage):
    name = "buffer_chunk"

    def run_writer(self, *, state: MutableMapping[str, Any], owner) -> None:
        _ = owner
        if bool(state.get("streaming", False)):
            return
        buffered = state.get("buffered_chunks")
        if not isinstance(buffered, list):
            buffered = []
            state["buffered_chunks"] = buffered
        buffered.append(
            {
                "src_path": state["src_path"],
                "num_rows": int(state["num_rows"]),
                "prediction_event_ids_np": np.asarray(state["prediction_event_ids_np"], dtype=np.int64),
                "prediction_columns": {k: np.asarray(v) for k, v in dict(state["prediction_columns"]).items()},
                "time_group_event_ids_np": (
                    None
                    if state.get("time_group_event_ids_np") is None
                    else np.asarray(state["time_group_event_ids_np"], dtype=np.int64)
                ),
                "time_group_ids_np": (
                    None
                    if state.get("time_group_ids_np") is None
                    else np.asarray(state["time_group_ids_np"], dtype=np.int64)
                ),
                "value_types": state.get("value_types"),
                "group_id_column": str(state.get("group_id_column", "time_group_ids")),
            }
        )

