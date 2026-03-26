from __future__ import annotations

import time
from collections import defaultdict
from typing import Any


class BaseDiagnostics:
    """Generic diagnostics accumulator for staged runtime."""

    def __init__(self, *, runtime_kind: str = "staged_runtime") -> None:
        self.runtime_kind = str(runtime_kind)
        self.stage_ms_total: dict[str, float] = defaultdict(float)
        self.stage_calls: dict[str, int] = defaultdict(int)
        self.stage_errors: dict[str, int] = defaultdict(int)
        self.rss_peak_mb = 0.0
        self.vram_peak_mb = 0.0
        self._t0 = time.perf_counter()

    def elapsed_s(self) -> float:
        return max(0.0, float(time.perf_counter() - self._t0))

    def record_stage_ms(self, *, stage_name: str, elapsed_ms: float) -> None:
        key = str(stage_name)
        self.stage_ms_total[key] += max(0.0, float(elapsed_ms))
        self.stage_calls[key] += 1

    def record_stage_error(self, *, stage_name: str) -> None:
        self.stage_errors[str(stage_name)] += 1

    def update_memory(self, *, rss_mb: float | None = None, vram_mb: float | None = None) -> None:
        if rss_mb is not None:
            self.rss_peak_mb = max(self.rss_peak_mb, float(rss_mb))
        if vram_mb is not None:
            self.vram_peak_mb = max(self.vram_peak_mb, float(vram_mb))

    def summary(self) -> dict[str, Any]:
        return {
            "runtime_kind": self.runtime_kind,
            "elapsed_s": self.elapsed_s(),
            "rss_peak_mb": float(self.rss_peak_mb),
            "vram_peak_mb": float(self.vram_peak_mb),
            "stage_ms_total": {k: float(v) for k, v in self.stage_ms_total.items()},
            "stage_calls": {k: int(v) for k, v in self.stage_calls.items()},
            "stage_errors": {k: int(v) for k, v in self.stage_errors.items()},
        }
