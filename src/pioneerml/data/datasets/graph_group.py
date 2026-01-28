"""GraphGroupDataset and supporting record dataclass aligned to mixed-event format."""

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
    hit_mask: Optional[Sequence[bool]] = None
    time_group_ids: Optional[Sequence[int]] = None
    labels: Optional[Sequence[int]] = None
    event_id: Optional[int] = None
    group_id: Optional[int] = None
    hit_labels: Optional[Sequence[Sequence[int]]] = None
    group_probs: Optional[Sequence[float]] = None
    hit_pdgs: Optional[Sequence[int]] = None
    class_energies: Optional[Sequence[float]] = None
    true_start: Optional[Sequence[float]] = None
    true_end: Optional[Sequence[float]] = None
    true_pion_stop: Optional[Sequence[float]] = None
    true_angle_vector: Optional[Sequence[float]] = None
    pred_pion_stop: Optional[Sequence[float]] = None
    pred_endpoints: Optional[Sequence[Sequence[Sequence[float]]]] = None
    matched_pion_index: Optional[int] = None
    pion_stop_for_angle: Optional[Sequence[float]] = None
    true_arc_length: Optional[float] = None

    def __getitem__(self, key):
        """Allow subscript-style access for backwards compatibility."""
        if not isinstance(key, str):
            raise TypeError(f"GraphRecord key must be a string, got {type(key)}")
        return getattr(self, key)


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
        node_features = torch.tensor(np.stack([coord, z_pos, energy, view], axis=1), dtype=torch.float)

        if item.hit_mask is not None:
            hit_mask = torch.tensor(item.hit_mask, dtype=torch.bool)
            if hit_mask.shape[0] != num_hits:
                raise ValueError("hit_mask length must match number of hits.")
        else:
            hit_mask = torch.ones(num_hits, dtype=torch.bool)
        num_valid = int(hit_mask.sum().item())

        if item.time_group_ids is not None:
            tg = torch.tensor(item.time_group_ids, dtype=torch.long)
            if tg.shape[0] != num_hits:
                raise ValueError("time_group_ids length must match number of hits.")
        else:
            tg = None

        edge_index = fully_connected_edge_index(num_valid, device=node_features.device)
        edge_attr = build_edge_attr(node_features[:num_valid], edge_index)

        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        data.hit_mask = hit_mask
        data.num_valid_hits = torch.tensor([num_valid], dtype=torch.long)
        if tg is not None:
            data.time_group_ids = tg

        # Add global group energy feature (shape [1, 1] for proper batching)
        data.u = torch.tensor([[node_features[:num_valid, 2].sum()]], dtype=torch.float)

        if item.labels is not None and self.num_classes:
            label_tensor = torch.zeros(self.num_classes, dtype=torch.float)
            for lbl in item.labels:
                if 0 <= lbl < self.num_classes:
                    label_tensor[lbl] = 1.0
            data.y_group = label_tensor.unsqueeze(0)
            data.y = label_tensor

        if item.hit_pdgs is not None:
            data.y_node = torch.tensor(item.hit_pdgs, dtype=torch.long)

        if item.class_energies is not None:
            data.y_energy = torch.tensor(item.class_energies, dtype=torch.float).unsqueeze(0)  # [1, num_classes]

        if item.hit_labels is not None:
            # Multi-label targets for splitter [N, 3]
            data.y = torch.tensor(item.hit_labels, dtype=torch.float)

        if item.event_id is not None:
            data.event_id = torch.tensor(int(item.event_id), dtype=torch.long)
        if item.group_id is not None:
            data.group_id = torch.tensor(int(item.group_id), dtype=torch.long)

        if item.true_start is not None and item.true_end is not None:
            # shape: [2, 3]
            start = torch.tensor(item.true_start, dtype=torch.float)
            end = torch.tensor(item.true_end, dtype=torch.float)
            data.y_pos = torch.stack([start, end], dim=0).unsqueeze(0)
            data.group_id = torch.tensor(int(item.group_id), dtype=torch.long)

        if item.true_pion_stop is not None:
            # shape: [1, 3]
            data.y_pion_stop = torch.tensor(item.true_pion_stop, dtype=torch.float).unsqueeze(0)

        if item.true_angle_vector is not None:
            # shape: [1, 3]
            data.y_angle_vector = torch.tensor(item.true_angle_vector, dtype=torch.float).unsqueeze(0)

        if item.pred_pion_stop is not None:
            # shape: [1, 3]
            data.pred_pion_stop = torch.tensor(item.pred_pion_stop, dtype=torch.float).unsqueeze(0)

        if item.group_probs is not None:
            data.group_probs = torch.tensor(item.group_probs, dtype=torch.float).unsqueeze(0)

        if item.true_arc_length is not None:
            data.y_arc = torch.tensor([item.true_arc_length], dtype=torch.float).unsqueeze(0)

        return data

    @staticmethod
    def _coerce(raw: Dict[str, Any] | GraphRecord) -> GraphRecord:
        # Fast path for same-class instance
        if isinstance(raw, GraphRecord):
            return raw

        # Duck typing for stale instances (from previous reloads)
        if hasattr(raw, 'coord'):
            return raw

        return GraphRecord(
            coord=raw["coord"],
            z=raw["z"],
            energy=raw["energy"],
            view=raw["view"],
            hit_mask=raw.get("hit_mask"),
            time_group_ids=raw.get("time_group_ids"),
            labels=raw.get("labels"),
            event_id=raw.get("event_id"),
            group_id=raw.get("group_id"),
            hit_pdgs=raw.get("hit_pdgs"),
            class_energies=raw.get("class_energies"),
            hit_labels=raw.get("hit_labels"),
            true_pion_stop=raw.get("true_pion_stop"),
            true_angle_vector=raw.get("true_angle_vector"),
            pred_pion_stop=raw.get("pred_pion_stop"),
            matched_pion_index=raw.get("matched_pion_index"),
            pion_stop_for_angle=raw.get("pion_stop_for_angle"),
            group_probs=raw.get("group_probs"),
            true_arc_length=raw.get("true_arc_length"),
            true_start=raw.get("true_start"),
            true_end=raw.get("true_end"),
            pred_endpoints=raw.get("pred_endpoints"),
        )
