from __future__ import annotations

from pathlib import Path

import numpy as np

from pioneerml.common.parquet import ParquetChunkReader

from ..base_loader import BaseLoader


class ParquetLoader(BaseLoader):
    """Loader constrained to parquet chunked inputs."""

    def __init__(
        self,
        *,
        parquet_paths: list[str],
        mode: str | None = None,
        batch_size: int = 64,
        row_groups_per_chunk: int = 4,
        num_workers: int = 0,
        input_columns: list[str] | None = None,
        target_columns: list[str] | None = None,
        columns: list[str] | None = None,
        split: str | None = None,
        train_fraction: float = 0.9,
        val_fraction: float = 0.05,
        test_fraction: float = 0.05,
        split_seed: int = 0,
        sample_fraction: float | None = None,
    ) -> None:
        super().__init__(batch_size=batch_size, num_workers=num_workers, mode=mode)
        resolved = [str(p) for p in parquet_paths]
        if not resolved:
            raise RuntimeError("No parquet paths provided.")
        missing = [p for p in resolved if not Path(p).exists()]
        if missing:
            raise RuntimeError(f"Missing parquet path(s): {missing}")

        self.parquet_paths = resolved
        self.row_groups_per_chunk = max(1, int(row_groups_per_chunk))
        self.input_columns = list(getattr(self, "input_columns", []) if input_columns is None else input_columns)
        self.target_columns = list(getattr(self, "target_columns", []) if target_columns is None else target_columns)
        self.columns = list(columns) if columns is not None else self.required_columns(include_targets=self.include_targets)
        split_norm = None if split is None else str(split).strip().lower()
        if split_norm is not None and split_norm not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}. Expected one of: 'train', 'val', 'test'.")
        self.split = split_norm
        self.train_fraction = float(train_fraction)
        self.val_fraction = float(val_fraction)
        self.test_fraction = float(test_fraction)
        total_frac = self.train_fraction + self.val_fraction + self.test_fraction
        if self.split is not None and not np.isclose(total_frac, 1.0, atol=1e-9):
            raise ValueError(
                "train_fraction + val_fraction + test_fraction must sum to 1.0 "
                f"(got {self.train_fraction} + {self.val_fraction} + {self.test_fraction} = {total_frac})."
            )
        self.split_seed = int(split_seed)
        self.sample_fraction = None if sample_fraction is None else float(sample_fraction)
        if self.sample_fraction is not None and not (0.0 < self.sample_fraction <= 1.0):
            raise ValueError(f"sample_fraction must be in (0, 1], got: {self.sample_fraction}")
        self.edge_populate_graph_block = 512

    def required_input_columns(self) -> list[str]:
        return list(self.input_columns)

    def required_target_columns(self) -> list[str]:
        return list(self.target_columns)

    def required_columns(self, *, include_targets: bool | None = None) -> list[str]:
        use_targets = self.include_targets if include_targets is None else bool(include_targets)
        cols = [*self.required_input_columns()]
        if use_targets:
            cols.extend(self.required_target_columns())
        return list(dict.fromkeys(cols))

    def _iter_tables(self):
        reader = ParquetChunkReader(
            parquet_paths=self.parquet_paths,
            columns=self.columns,
            row_groups_per_chunk=self.row_groups_per_chunk,
        )
        yield from reader.iter_tables()
