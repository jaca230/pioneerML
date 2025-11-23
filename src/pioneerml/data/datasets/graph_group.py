"""
GraphGroupDataset and supporting record dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Optional, List, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from pioneerml.data.datasets.utils import fully_connected_edge_index, build_edge_attr


@dataclass
class GraphRecord:
    coord: Iterable[float]
    z: Iterable[float]
    energy: Iterable[float]
    view: Iterable[float]
    labels: Optional[Sequence[int]] = None
    event_id: Optional[int] = None
    group_id: Optional[int] = None
    hit_labels: Optional[Sequence[Sequence[int]]] = None
    group_probs: Optional[Sequence[float]] = None


class GraphGroupDataset(Dataset):
    """Dataset that emits standardized graph Data objects for time-group records."""

    def __init__(self, records: Sequence[Dict[str, Any] | GraphRecord], *, num_classes: Optional[int] = None):
        self.items: List[GraphRecord] = [self._coerce(item) for item in records]
        if num_classes is None:
            max_label = -1
            for item in self.items:
                if item.labels:
                    max_label = max(max_label, max(item.labels))
            num_classes = max_label + 1 if max_label >= 0 else 0
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Data:
        item = self.items[index]

        coord = np.asarray(item.coord, dtype=np.float32)
        z_pos = np.asarray(item.z, dtype=np.float32)
        energy = np.asarray(item.energy, dtype=np.float32)
        view = np.asarray(item.view, dtype=np.float32)

        if not (coord.shape == z_pos.shape == energy.shape == view.shape):
            raise ValueError("All per-hit arrays must share the same shape.")

        num_hits = coord.shape[0]
        group_energy = np.full(num_hits, energy.sum(), dtype=np.float32)
        node_features = torch.tensor(
            np.stack([coord, z_pos, energy, view, group_energy], axis=1), dtype=torch.float
        )

        edge_index = fully_connected_edge_index(num_hits, device=node_features.device)
        edge_attr = build_edge_attr(node_features, edge_index)

        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

        if item.labels and self.num_classes:
            label_tensor = torch.zeros(self.num_classes, dtype=torch.float)
            for lbl in item.labels:
                if 0 <= lbl < self.num_classes:
                    label_tensor[lbl] = 1.0
            data.y = label_tensor

        if item.event_id is not None:
            data.event_id = torch.tensor(int(item.event_id), dtype=torch.long)
        if item.group_id is not None:
            data.group_id = torch.tensor(int(item.group_id), dtype=torch.long)

        # Preserve original record for downstream evaluation/visualization
        data._raw = item

        return data

    @staticmethod
    def _coerce(raw: Dict[str, Any] | GraphRecord) -> GraphRecord:
        if isinstance(raw, GraphRecord):
            return raw
        return GraphRecord(
            coord=raw["coord"],
            z=raw["z"],
            energy=raw["energy"],
            view=raw["view"],
            labels=raw.get("labels"),
            event_id=raw.get("event_id"),
            group_id=raw.get("group_id"),
        )
