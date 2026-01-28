from __future__ import annotations

from typing import Optional

from pioneerml.training.datamodules import GraphDataModule
from .processor import GroupClassificationProcessor
from .loader import GroupClassificationLoader
from .dataset import GroupClassificationPolarsDataset


def build_datamodule(
    parquet_pattern: str,
    *,
    max_files: int | None = None,
    limit_groups: int | None = None,
    batch_size: int = 128,
    num_workers: int = 0,
    val_split: float = 0.15,
    test_split: float = 0.0,
    seed: int = 42,
    max_hits: int = 256,
    pad_value: float = 0.0,
    compute_time_groups: bool = True,
    time_window_ns: float = 1.0,
) -> GraphDataModule:
    processor = GroupClassificationProcessor(
        max_hits=max_hits,
        pad_value=pad_value,
        compute_time_groups=compute_time_groups,
        time_window_ns=time_window_ns,
    )
    loader = GroupClassificationLoader(columns=processor.columns)
    df = loader.load(parquet_pattern, max_files=max_files, limit_groups=limit_groups)
    dataset = GroupClassificationPolarsDataset(
        df,
        max_hits=max_hits,
        pad_value=pad_value,
        compute_time_groups=compute_time_groups,
        time_window_ns=time_window_ns,
    )

    dm = GraphDataModule(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
    )
    dm.setup(stage="fit")
    return dm
