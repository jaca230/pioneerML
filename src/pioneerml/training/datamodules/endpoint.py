"""DataModule for endpoint regression."""

from __future__ import annotations

from typing import Optional, Sequence

from pioneerml.data import EndpointGraphDataset
from pioneerml.training.datamodules.base import GraphDataModule


class EndpointDataModule(GraphDataModule):
    """DataModule for endpoint regression."""

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
        num_quantiles: int = 3,
    ):
        dataset = EndpointGraphDataset(records, num_quantiles=num_quantiles)
        super().__init__(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
        )
