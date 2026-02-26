from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from ..stage_context import StageContext
from ..utils.loader_diagnostics import LoaderDiagnostics
from .base import StageObserver


class MemoryObserver(StageObserver):
    """RSS/VRAM observer sampled per stage."""

    def __init__(self, *, diagnostics: LoaderDiagnostics, track_rss: bool = True, track_vram: bool = False) -> None:
        self.diagnostics = diagnostics
        self.track_rss = bool(track_rss)
        self.track_vram = bool(track_vram)
        self._psutil_process = None
        if self.track_rss:
            try:
                import psutil  # type: ignore

                self._psutil_process = psutil.Process()
            except Exception:
                self._psutil_process = None

    def _rss_mb(self) -> float | None:
        if self._psutil_process is None:
            return None
        try:
            return float(self._psutil_process.memory_info().rss) / (1024.0 * 1024.0)
        except Exception:
            return None

    def _vram_mb(self) -> float | None:
        if not self.track_vram:
            return None
        try:
            import torch

            if not torch.cuda.is_available():
                return None
            return float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
        except Exception:
            return None

    def after_stage(self, *, context: StageContext, state: MutableMapping[str, Any], loader: Any) -> None:
        _ = context
        _ = state
        _ = loader
        self.diagnostics.update_memory(rss_mb=self._rss_mb(), vram_mb=self._vram_mb())

    def on_chunk_end(self, *, chunk_index: int, state: MutableMapping[str, Any], loader: Any) -> None:
        _ = chunk_index
        _ = state
        _ = loader
        self.diagnostics.update_memory(rss_mb=self._rss_mb(), vram_mb=self._vram_mb())
