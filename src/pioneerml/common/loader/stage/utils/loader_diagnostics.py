from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Mapping
from typing import Any


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


class LoaderDiagnostics:
    """Aggregated diagnostics for staged loader execution."""

    def __init__(self, *, loader_kind: str = "loader") -> None:
        self.loader_kind = str(loader_kind)
        self.stage_ms_total: dict[str, float] = defaultdict(float)
        self.stage_calls: dict[str, int] = defaultdict(int)
        self.stage_errors: dict[str, int] = defaultdict(int)
        self.rows_total = 0
        self.graphs_total = 0
        self.nodes_total = 0
        self.edges_total = 0
        self.chunk_graphs_total = 0
        self.chunk_nodes_total = 0
        self.chunk_edges_total = 0
        self.batches_total = 0
        self.chunks_total = 0
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

    def record_chunk(self, *, raw_num_rows: int, state: Mapping[str, Any]) -> None:
        self.chunks_total += 1
        self.rows_total += max(0, int(raw_num_rows))
        chunk_out = state.get("chunk_out")
        if isinstance(chunk_out, Mapping):
            self.chunk_graphs_total += _safe_int(chunk_out.get("num_graphs"), 0)
            x = chunk_out.get("x")
            edge_index = chunk_out.get("edge_index")
            self.chunk_nodes_total += _safe_int(getattr(x, "shape", [0])[0] if x is not None else 0, 0)
            self.chunk_edges_total += _safe_int(getattr(edge_index, "shape", [0, 0])[1] if edge_index is not None else 0, 0)

    def record_batch(self, *, batch: Any) -> None:
        self.batches_total += 1
        self.graphs_total += _safe_int(getattr(batch, "num_graphs", 0), 0)
        self.nodes_total += _safe_int(getattr(getattr(batch, "x", None), "shape", [0])[0], 0)
        self.edges_total += _safe_int(getattr(getattr(batch, "edge_index", None), "shape", [0, 0])[1], 0)

    def update_memory(self, *, rss_mb: float | None = None, vram_mb: float | None = None) -> None:
        if rss_mb is not None:
            self.rss_peak_mb = max(self.rss_peak_mb, float(rss_mb))
        if vram_mb is not None:
            self.vram_peak_mb = max(self.vram_peak_mb, float(vram_mb))

    def summary(self) -> dict[str, Any]:
        elapsed = self.elapsed_s()
        rows_per_s = (self.rows_total / elapsed) if elapsed > 0.0 else 0.0
        graphs_per_s = (self.graphs_total / elapsed) if elapsed > 0.0 else 0.0
        nodes_per_s = (self.nodes_total / elapsed) if elapsed > 0.0 else 0.0
        edges_per_s = (self.edges_total / elapsed) if elapsed > 0.0 else 0.0
        return {
            "loader_kind": self.loader_kind,
            "elapsed_s": elapsed,
            "chunks_total": int(self.chunks_total),
            "rows_total": int(self.rows_total),
            "graphs_total": int(self.graphs_total),
            "nodes_total": int(self.nodes_total),
            "edges_total": int(self.edges_total),
            "chunk_graphs_total": int(self.chunk_graphs_total),
            "chunk_nodes_total": int(self.chunk_nodes_total),
            "chunk_edges_total": int(self.chunk_edges_total),
            "batches_total": int(self.batches_total),
            "rows_per_s": float(rows_per_s),
            "graphs_per_s": float(graphs_per_s),
            "nodes_per_s": float(nodes_per_s),
            "edges_per_s": float(edges_per_s),
            "rss_peak_mb": float(self.rss_peak_mb),
            "vram_peak_mb": float(self.vram_peak_mb),
            "stage_ms_total": {k: float(v) for k, v in self.stage_ms_total.items()},
            "stage_calls": {k: int(v) for k, v in self.stage_calls.items()},
            "stage_errors": {k: int(v) for k, v in self.stage_errors.items()},
        }
