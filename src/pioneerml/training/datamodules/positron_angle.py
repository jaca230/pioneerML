"""
DataModule for positron angle regression datasets.
"""

from __future__ import annotations

from typing import Optional, Sequence

from pioneerml.data import PositronAngleDataset
from pioneerml.training.datamodules.base import GraphDataModule


class PositronAngleDataModule(GraphDataModule):
    """DataModule for positron angle regression datasets."""

    def __init__(
        self,
        records: Sequence,
        *,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        val_split: float = 0.2,
        test_split: float = 0.0,
        seed: int = 42,
    ):
        dataset = PositronAngleDataset(records)
        super().__init__(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
        )
