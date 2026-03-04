from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

from .base_stage import BaseWriterStage


class OpenSinksStage(BaseWriterStage):
    name = "open_sinks"

    def run_writer(self, *, state: MutableMapping[str, Any], owner) -> None:
        writer = owner
        if not bool(state.get("streaming", False)):
            return

        source_contexts = list(state.get("source_contexts") or [])
        output_dir = Path(state["output_dir"])
        output_path = state.get("output_path")
        sink_entries: dict[str, dict[str, Any]] = {}

        for idx, ctx in enumerate(source_contexts):
            src_path = Path(ctx["src_path"]).expanduser().resolve()
            scoped_output_path = str(output_path) if (output_path and len(source_contexts) == 1 and idx == 0) else None
            pred_path = writer.resolve_prediction_output_path(
                src_path=src_path,
                output_dir=output_dir,
                output_path=scoped_output_path,
            )
            sink_entries[str(src_path)] = {
                "sink": writer.output_backend.open_sink(dst_path=pred_path),
                "pred_path": pred_path,
                "src_path": src_path,
            }

        state["open_stream_sinks"] = sink_entries

