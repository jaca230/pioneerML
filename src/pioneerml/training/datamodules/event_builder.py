"""
DataModule for EventBuilder training on mixed events.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split

from pioneerml.data.event_mixer import MixedEventDataset, EventContainer
from pioneerml.data.datasets.utils import build_event_graph


def event_builder_collate(radius_z: float = 0.5):
    """Factory for collate_fn that builds block-diagonal batches for EventBuilder."""

    def _collate(containers: List[EventContainer]):
        batch_x = []
        batch_edge_index = []
        batch_edge_attr = []
        batch_group_indices = []
        batch_targets = []
        batch_indices_per_group = []

        node_offset = 0
        group_offset = 0
        valid_batch_idx = 0

        for container in containers:
            try:
                graph_data = build_event_graph(container, device=torch.device("cpu"), radius_z=radius_z)
            except Exception:
                continue
            if graph_data is None:
                continue

            x, edge_idx, edge_attr, groups, num_groups, targets = graph_data

            batch_x.append(x)
            batch_edge_attr.append(edge_attr)
            batch_targets.append(targets)

            batch_edge_index.append(edge_idx + node_offset)
            batch_group_indices.append(groups + group_offset)
            batch_indices_per_group.append(
                torch.full((num_groups,), valid_batch_idx, dtype=torch.long)
            )

            node_offset += x.size(0)
            group_offset += num_groups
            valid_batch_idx += 1

        if len(batch_x) == 0:
            return None

        big_x = torch.cat(batch_x, dim=0)
        big_edge_index = torch.cat(batch_edge_index, dim=1)
        big_edge_attr = torch.cat(batch_edge_attr, dim=0)
        big_group_indices = torch.cat(batch_group_indices, dim=0)
        big_batch_indices = torch.cat(batch_indices_per_group, dim=0)
        big_targets = torch.block_diag(*batch_targets)

        return big_x, big_edge_index, big_edge_attr, big_group_indices, big_batch_indices, big_targets

    return _collate


class EventBuilderDataModule:
    """
    Lightweight DataModule for EventBuilder training.

    Expects one or more mixed-event .pt files produced by EventMixer/save_mixed_events.
    """

    def __init__(
        self,
        mixed_paths: Sequence[str | Path],
        *,
        val_split: float = 0.1,
        batch_size: int = 32,
        num_workers: int = 0,
        radius_z: float = 0.5,
        seed: int = 42,
    ):
        paths = [Path(p) for p in mixed_paths]
        datasets = [MixedEventDataset(str(p)) for p in paths]
        self.full_dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.radius_z = radius_z
        self.seed = seed

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        if self.val_split <= 0 or len(self.full_dataset) < 2:
            self.train_dataset = self.full_dataset
            self.val_dataset = None
            return
        total_len = len(self.full_dataset)
        val_len = max(1, int(total_len * self.val_split))
        train_len = total_len - val_len
        self.train_dataset, self.val_dataset = random_split(
            self.full_dataset,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(self.seed),
        )

    def _make_loader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=event_builder_collate(self.radius_z),
        )

    def train_dataloader(self):
        return self._make_loader(self.train_dataset)

    def val_dataloader(self):
        if self.val_dataset is None:
            return []
        return self._make_loader(self.val_dataset)

    def test_dataloader(self):
        return self.val_dataloader()
