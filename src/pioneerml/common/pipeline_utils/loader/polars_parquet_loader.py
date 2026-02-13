from __future__ import annotations

from pathlib import Path

import polars as pl


class PolarsParquetLoader:
    """Generic parquet ingestion helper using Polars."""

    def load_rows(self, *, parquet_paths: list[str], columns: list[str]) -> list[dict]:
        paths = [str(Path(p).expanduser().resolve()) for p in parquet_paths]
        if not paths:
            return []
        frame = pl.scan_parquet(paths).select(columns).collect(streaming=True)
        return frame.to_dicts()
