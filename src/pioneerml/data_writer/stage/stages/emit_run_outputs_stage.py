from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from .base_stage import BaseWriterStage


class EmitRunOutputsStage(BaseWriterStage):
    name = "emit_run_outputs"

    def run_writer(self, *, state: MutableMapping[str, Any], owner) -> None:
        _ = owner
        paths = list(dict.fromkeys(str(p) for p in list(state.get("written_prediction_paths") or [])))
        ts_paths = list(dict.fromkeys(str(p) for p in list(state.get("written_timestamped_paths") or [])))
        state["run_outputs"] = {
            "predictions_path": paths[0] if len(paths) == 1 else None,
            "predictions_paths": paths,
            "timestamped_predictions_path": ts_paths[0] if len(ts_paths) == 1 else None,
            "timestamped_predictions_paths": ts_paths,
        }
