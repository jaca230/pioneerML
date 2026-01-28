from __future__ import annotations

import glob
from pathlib import Path
from typing import Optional, Sequence

import polars as pl


class GroupClassificationLoader:
    """Pipeline-specific loader that reads only required columns."""

    def __init__(self, columns: Sequence[str]):
        self.columns = list(columns)

    def load(
        self,
        parquet_pattern: str,
        *,
        max_files: Optional[int] = None,
        limit_groups: Optional[int] = None,
    ) -> pl.DataFrame:
        paths = sorted([Path(p) for p in glob.glob(parquet_pattern)])
        if not paths:
            raise FileNotFoundError(f"No files matched pattern '{parquet_pattern}'")
        if max_files is not None:
            paths = paths[:max_files]

        dfs = []
        remaining = limit_groups
        for path in paths:
            df = pl.read_parquet(path, columns=self.columns)
            if remaining is not None:
                df = df.head(remaining)
                remaining -= df.height
            dfs.append(df)
            if remaining is not None and remaining <= 0:
                break

        if dfs:
            return pl.concat(dfs, how="vertical")
        return pl.DataFrame({c: [] for c in self.columns})
