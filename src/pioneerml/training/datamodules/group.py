"""
DataModule for group-level classification datasets.
"""

from __future__ import annotations

from typing import Optional, Sequence

from pioneerml.data import GraphGroupDataset
from pioneerml.training.datamodules.base import GraphDataModule


class GroupClassificationDataModule(GraphDataModule):
    """DataModule for group-level classification datasets."""

    def __init__(
        self,
        records: Sequence,
        *,
        num_classes: Optional[int] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        val_split: float = 0.2,
        test_split: float = 0.0,
        seed: int = 42,
    ):
        dataset = GraphGroupDataset(records, num_classes=num_classes)
        super().__init__(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
        )
