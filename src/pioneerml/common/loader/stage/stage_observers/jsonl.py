from __future__ import annotations

import json
import time
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

from ..stage_context import StageContext
from .base import StageObserver


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


class JsonlObserver(StageObserver):
    """Structured event logger for stages/chunks."""

    def __init__(self, *, path: str | Path, append: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        self._fh = self.path.open(mode, encoding="utf-8")
        self._start_ns: dict[tuple[int, int], int] = {}

    def _write(self, payload: Mapping[str, Any]) -> None:
        self._fh.write(json.dumps(dict(payload), ensure_ascii=True) + "\n")
        self._fh.flush()

    def before_stage(self, *, context: StageContext, state: MutableMapping[str, Any], loader: Any) -> None:
        _ = state
        _ = loader
        self._start_ns[(context.chunk_index, context.stage_index)] = time.perf_counter_ns()

    def after_stage(self, *, context: StageContext, state: MutableMapping[str, Any], loader: Any) -> None:
        _ = loader
        key = (context.chunk_index, context.stage_index)
        t0 = self._start_ns.pop(key, None)
        elapsed_ms = ((time.perf_counter_ns() - t0) / 1_000_000.0) if t0 is not None else None
        self._write(
            {
                "event": "stage_end",
                "chunk_index": int(context.chunk_index),
                "stage_index": int(context.stage_index),
                "stage_name": context.stage_name,
                "raw_num_rows": int(context.raw_num_rows),
                "elapsed_ms": _safe_float(elapsed_ms, 0.0) if elapsed_ms is not None else None,
                "stop_pipeline": bool(state.get("stop_pipeline", False)),
                "has_chunk_out": state.get("chunk_out") is not None,
            }
        )

    def on_error(
        self,
        *,
        context: StageContext,
        state: MutableMapping[str, Any],
        loader: Any,
        error: Exception,
    ) -> None:
        _ = state
        _ = loader
        self._write(
            {
                "event": "stage_error",
                "chunk_index": int(context.chunk_index),
                "stage_index": int(context.stage_index),
                "stage_name": context.stage_name,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )

    def on_chunk_end(self, *, chunk_index: int, state: MutableMapping[str, Any], loader: Any) -> None:
        _ = loader
        chunk_out = state.get("chunk_out")
        num_graphs = _safe_int(chunk_out.get("num_graphs"), 0) if isinstance(chunk_out, Mapping) else 0
        self._write(
            {
                "event": "chunk_end",
                "chunk_index": int(chunk_index),
                "raw_num_rows": _safe_int(state.get("raw_num_rows"), 0),
                "num_graphs": int(num_graphs),
                "stop_pipeline": bool(state.get("stop_pipeline", False)),
            }
        )
