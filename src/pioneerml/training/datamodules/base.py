"""
Base Lightning DataModule for graph datasets.
"""

from __future__ import annotations

from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, random_split
from torch_geometric.loader import DataLoader as GeoDataLoader


class GraphDataModule(pl.LightningDataModule):
    """Generic DataModule for graph datasets with optional train/val/test splits."""

    def __init__(
        self,
        dataset: Optional[Dataset] = None,
        *,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        val_split: float = 0.2,
        test_split: float = 0.0,
        seed: int = 42,
    ):
        super().__init__()
        if dataset is None and train_dataset is None:
            raise ValueError("Provide either a base dataset to split or explicit train_dataset.")
        if not 0 <= val_split < 1 or not 0 <= test_split < 1:
            raise ValueError("val_split and test_split must be in [0, 1).")
        if val_split + test_split >= 1:
            raise ValueError("val_split + test_split must be less than 1.")

        self._base_dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed

        self.train_dataset: Optional[Dataset] = train_dataset
        self.val_dataset: Optional[Dataset] = val_dataset
        self.test_dataset: Optional[Dataset] = test_dataset

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is not None:
            return

        if self._base_dataset is None:
            raise ValueError("No dataset available to split.")

        total = len(self._base_dataset)
        val_len = int(total * self.val_split)
        test_len = int(total * self.test_split)
        train_len = total - val_len - test_len

        if val_len == 0 and test_len == 0:
            self.train_dataset = self._base_dataset
            self.val_dataset = None
            self.test_dataset = None
            return

        lengths = [train_len, val_len, test_len] if test_len > 0 else [train_len, val_len]
        generator = torch.Generator().manual_seed(self.seed)
        splits = random_split(self._base_dataset, lengths, generator=generator)

        self.train_dataset = splits[0]
        self.val_dataset = splits[1] if len(splits) > 1 and val_len > 0 else None
        self.test_dataset = splits[2] if len(splits) > 2 and test_len > 0 else None

    def train_dataloader(self):
        return GeoDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return []
        return GeoDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return []
        return GeoDataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
