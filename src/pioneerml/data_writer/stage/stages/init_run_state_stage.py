from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from .base_stage import BaseWriterStage


class InitRunStateStage(BaseWriterStage):
    name = "init_run_state"

    def run_writer(self, *, state: MutableMapping[str, Any], owner) -> None:
        _ = owner
        state["written_prediction_paths"] = []
        state["written_timestamped_paths"] = []
        state["buffered_chunks"] = []

