from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

from .base_stage import BaseWriterStage


class AppendChunkStage(BaseWriterStage):
    name = "append_chunk"

    def run_writer(self, *, state: MutableMapping[str, Any], owner) -> None:
        writer = owner
        if not bool(state.get("streaming", False)):
            return

        table = state.get("table")
        if table is None:
            raise ValueError("Missing 'table' before append_chunk.")
        src_path = Path(state["src_path"]).expanduser().resolve()
        sink_entries = dict(state.get("open_stream_sinks") or {})
        entry = sink_entries.get(str(src_path))
        if entry is None:
            raise RuntimeError(f"No open sink entry for source '{src_path}'.")
        writer.output_backend.append_chunk(sink=entry["sink"], table=table)
