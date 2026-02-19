"""Shared PyG LightningDataModule for tutorial graph notebooks/pipelines."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader


class GraphDataModule(pl.LightningDataModule):
    """Minimal PyG datamodule used by tutorial examples."""

    def __init__(
        self,
        dataset: list[Data],
        *,
        val_split: float = 0.2,
        batch_size: int = 32,
        num_workers: int = 0,
        seed: int = 123,
    ):
        super().__init__()
        self.dataset = list(dataset)
        self.val_split = float(val_split)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.seed = int(seed)
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        if self.train_dataset is not None:
            return
        total = len(self.dataset)
        val_len = int(total * self.val_split)
        if total > 1:
            val_len = min(max(val_len, 1), total - 1)
        else:
            val_len = 0
        train_len = total - val_len
        splits = random_split(
            self.dataset,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(self.seed),
        )
        self.train_dataset, self.val_dataset = splits[0], splits[1]

    def train_dataloader(self):
        return PyGDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        if self.val_dataset is None or len(self.val_dataset) == 0:
            return []
        return PyGDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
