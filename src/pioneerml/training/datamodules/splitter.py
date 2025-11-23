"""
DataModule for hit-level splitter datasets.
"""

from __future__ import annotations

from typing import Sequence

from pioneerml.data import SplitterGraphDataset
from pioneerml.training.datamodules.base import GraphDataModule


class SplitterDataModule(GraphDataModule):
    """DataModule for hit-level splitter datasets."""

    def __init__(
        self,
        records: Sequence,
        *,
        use_group_probs: bool = False,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        val_split: float = 0.2,
        test_split: float = 0.0,
        seed: int = 42,
    ):
        dataset = SplitterGraphDataset(records, use_group_probs=use_group_probs)
        super().__init__(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
        )
