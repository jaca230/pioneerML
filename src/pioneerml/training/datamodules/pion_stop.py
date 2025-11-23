"""
DataModule for pion stop regression datasets.
"""

from __future__ import annotations

from typing import Sequence

from pioneerml.data import PionStopGraphDataset
from pioneerml.training.datamodules.base import GraphDataModule


class PionStopDataModule(GraphDataModule):
    """DataModule for pion stop regression datasets."""

    def __init__(
        self,
        records: Sequence,
        *,
        pion_pdg: int = 1,
        min_pion_hits: int = 1,
        use_true_time: bool = True,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        val_split: float = 0.2,
        test_split: float = 0.0,
        seed: int = 42,
    ):
        dataset = PionStopGraphDataset(
            records,
            pion_pdg=pion_pdg,
            min_pion_hits=min_pion_hits,
            use_true_time=use_true_time,
        )
        super().__init__(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
        )
