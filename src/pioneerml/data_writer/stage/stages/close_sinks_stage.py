from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

from .base_stage import BaseWriterStage


class CloseSinksStage(BaseWriterStage):
    name = "close_sinks"

    def run_writer(self, *, state: MutableMapping[str, Any], owner) -> None:
        writer = owner
        if not bool(state.get("streaming", False)):
            return

        sink_entries = dict(state.get("open_stream_sinks") or {})
        output_dir = Path(state["output_dir"])
        write_timestamped = bool(state.get("write_timestamped", False))
        timestamp = str(state.get("timestamp", writer.timestamp()))

        written = state.get("written_prediction_paths")
        if not isinstance(written, list):
            written = []
            state["written_prediction_paths"] = written
        written_ts = state.get("written_timestamped_paths")
        if not isinstance(written_ts, list):
            written_ts = []
            state["written_timestamped_paths"] = written_ts

        for entry in sink_entries.values():
            sink = entry["sink"]
            pred_path = Path(entry["pred_path"])
            src_path = Path(entry["src_path"])
            writer.output_backend.close_sink(sink=sink)
            written.append(str(pred_path))

            if write_timestamped:
                timestamped = output_dir / f"{src_path.stem}_preds_{timestamp}{writer.output_backend.default_extension()}"
                writer.write_timestamped_copy(pred_path, timestamped)
                written_ts.append(str(timestamped))

        state["open_stream_sinks"] = {}

