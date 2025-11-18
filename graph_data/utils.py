"""
Common utilities for building fully connected graph representations of
preprocessed time-group data and feeding them into torch-geometric models.

Provides:
  * fully_connected_edge_index
  * build_edge_attr
  * GraphGroupDataset
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Optional, List, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


def fully_connected_edge_index(num_nodes: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Return a directed fully-connected edge index without self loops."""
    if num_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    src = torch.arange(num_nodes, device=device).repeat_interleave(num_nodes - 1)
    dst = torch.cat([
        torch.cat([torch.arange(0, i, device=device), torch.arange(i + 1, num_nodes, device=device)])
        for i in range(num_nodes)
    ])
    return torch.stack([src, dst], dim=0)


def build_edge_attr(node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Compute edge attributes [dx, dz, dE, same_view] for provided edges."""
    if edge_index.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float, device=node_features.device)

    src, dst = edge_index
    coord = node_features[:, 0]
    z_pos = node_features[:, 1]
    energy = node_features[:, 2]
    view_flag = node_features[:, 3]

    dx = (coord[dst] - coord[src]).unsqueeze(1)
    dz = (z_pos[dst] - z_pos[src]).unsqueeze(1)
    dE = (energy[dst] - energy[src]).unsqueeze(1)
    same_view = (view_flag[dst] == view_flag[src]).float().unsqueeze(1)

    return torch.cat([dx, dz, dE, same_view], dim=1)


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


@dataclass
class PionStopRecord:
    coord: Iterable[float]
    z: Iterable[float]
    energy: Iterable[float]
    view: Iterable[float]
    time: Iterable[float]
    pdg: Iterable[int]
    true_x: Iterable[float]
    true_y: Iterable[float]
    true_z: Iterable[float]
    true_time: Iterable[float]
    event_id: Optional[int] = None
    group_id: Optional[int] = None


class PionStopGraphDataset(Dataset):
    """
    Dataset for regressing pion stop positions from time-group graphs.

    Each record must provide per-hit true coordinates (true_x/true_y/true_z),
    particle identifiers (pdg), and truth timing information. The target is
    derived from the final pion hit within the group.
    """

    def __init__(
        self,
        records: Sequence[PionStopRecord | Dict[str, Any]],
        *,
        pion_pdg: int = 1,
        min_pion_hits: int = 1,
        use_true_time: bool = True,
    ):
        self.items: List[PionStopRecord] = [self._coerce(item) for item in records]
        self.pion_pdg = pion_pdg
        self.min_pion_hits = max(1, min_pion_hits)
        self.use_true_time = use_true_time

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

        pdg = np.asarray(item.pdg, dtype=np.int32)
        true_x = np.asarray(item.true_x, dtype=np.float32)
        true_y = np.asarray(item.true_y, dtype=np.float32)
        true_z = np.asarray(item.true_z, dtype=np.float32)
        true_time = np.asarray(item.true_time, dtype=np.float32)
        hit_time = np.asarray(item.time, dtype=np.float32)

        if not (
            pdg.shape == true_x.shape == true_y.shape == true_z.shape == true_time.shape == hit_time.shape == coord.shape
        ):
            raise ValueError("All PionStopRecord arrays must align per hit.")

        pion_indices = np.flatnonzero(pdg == self.pion_pdg)
        if pion_indices.size < self.min_pion_hits:
            print(pdg, self.pion_pdg)
            raise ValueError("Record does not contain enough pion hits to compute stop target.")

        if self.use_true_time:
            ref_time = true_time[pion_indices]
        else:
            ref_time = hit_time[pion_indices]
        last_idx = pion_indices[int(np.argmax(ref_time))]
        stop_target = np.array([true_x[last_idx], true_y[last_idx], true_z[last_idx]], dtype=np.float32)

        num_hits = coord.shape[0]
        group_energy = np.full(num_hits, energy.sum(), dtype=np.float32)
        node_features = torch.tensor(
            np.stack([coord, z_pos, energy, view, group_energy], axis=1),
            dtype=torch.float,
        )

        edge_index = fully_connected_edge_index(num_hits, device=node_features.device)
        edge_attr = build_edge_attr(node_features, edge_index)

        target_tensor = torch.tensor(stop_target, dtype=torch.float).unsqueeze(0)

        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=target_tensor,
        )

        if item.event_id is not None:
            data.event_id = torch.tensor(int(item.event_id), dtype=torch.long)
        if item.group_id is not None:
            data.group_id = torch.tensor(int(item.group_id), dtype=torch.long)

        return data

    @staticmethod
    def _coerce(raw: PionStopRecord | Dict[str, Any]) -> PionStopRecord:
        if isinstance(raw, PionStopRecord):
            return raw
        return PionStopRecord(
            coord=raw["coord"],
            z=raw["z"],
            energy=raw["energy"],
            view=raw["view"],
            time=raw["time"],
            pdg=raw["pdg"],
            true_x=raw["true_x"],
            true_y=raw["true_y"],
            true_z=raw["true_z"],
            true_time=raw["true_time"],
            event_id=raw.get("event_id"),
            group_id=raw.get("group_id"),
        )

class SplitterGraphDataset(Dataset):
    """
    Dataset for the splitter network.

    - Uses the same standardized node features as GraphGroupDataset:
      [coord, z, energy, view, group_energy]
    - Expects per-hit multi-label targets in GraphRecord.hit_labels
      with shape [num_hits, 3] corresponding to [is_pion, is_muon, is_mip].
    - Optionally appends group-level classifier probabilities
      [p_pi, p_mu, p_mip] to each node's feature vector.
    """

    def __init__(
        self,
        records: Sequence[GraphRecord | dict],
        *,
        use_group_probs: bool = False,
    ):
        # Normalize to GraphRecord
        self.items: list[GraphRecord] = [self._coerce(item) for item in records]
        self.use_group_probs = use_group_probs

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

        # Base node features: identical to GraphGroupDataset
        base_features = np.stack(
            [coord, z_pos, energy, view, group_energy],
            axis=1,
        )  # [N, 5]

        # Optional classifier probabilities [p_pi, p_mu, p_mip]
        if self.use_group_probs and item.group_probs is not None:
            probs = np.asarray(item.group_probs, dtype=np.float32)  # [3]
            if probs.shape != (3,):
                raise ValueError(f"group_probs must have shape (3,), got {probs.shape}")
            probs_expanded = np.repeat(probs[None, :], num_hits, axis=0)  # [N, 3]
            node_features = np.concatenate([base_features, probs_expanded], axis=1)  # [N, 8]
        else:
            node_features = base_features  # [N, 5]

        x = torch.tensor(node_features, dtype=torch.float)

        # Per-hit multi-label targets: [N, 3] of 0/1
        if item.hit_labels is None:
            raise ValueError("SplitterGraphDataset requires GraphRecord.hit_labels for each record.")

        labels_arr = np.asarray(item.hit_labels, dtype=np.float32)
        if labels_arr.shape[0] != num_hits:
            raise ValueError(
                f"hit_labels length {labels_arr.shape[0]} does not match number of hits {num_hits}"
            )
        if labels_arr.ndim != 2 or labels_arr.shape[1] != 3:
            raise ValueError(
                f"hit_labels must have shape [num_hits, 3] (pion, muon, mip), got {labels_arr.shape}"
            )

        y = torch.tensor(labels_arr, dtype=torch.float)  # [N, 3]

        edge_index = fully_connected_edge_index(num_hits, device=x.device)
        edge_attr = build_edge_attr(x, edge_index)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        if item.event_id is not None:
            data.event_id = torch.tensor(int(item.event_id), dtype=torch.long)
        if item.group_id is not None:
            data.group_id = torch.tensor(int(item.group_id), dtype=torch.long)

        return data

    @staticmethod
    def _coerce(raw: GraphRecord | dict) -> GraphRecord:
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
            hit_labels=raw.get("hit_labels"),
            group_probs=raw.get("group_probs"),
        )
