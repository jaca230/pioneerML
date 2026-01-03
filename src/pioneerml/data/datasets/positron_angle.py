"""
Graph dataset for positron angle regression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Dict, Any, List

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from pioneerml.data.datasets.utils import fully_connected_edge_index, build_edge_attr


@dataclass
class PositronAngleRecord:
    coord: Iterable[float]
    z: Iterable[float]
    energy: Iterable[float]
    view: Iterable[float]
    angle: Sequence[float] | None = None  # target regression vector length 2
    event_id: Optional[int] = None
    group_id: Optional[int] = None
    pion_stop: Optional[Sequence[float]] = None
    true_angle_vector: Optional[Sequence[float]] = None


class PositronAngleDataset(Dataset):
    """Dataset emitting graph Data objects with angle regression targets."""

    def __init__(self, records: Sequence[Dict[str, Any] | PositronAngleRecord]):
        self.items: List[PositronAngleRecord] = [self._coerce(r) for r in records]

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

        edge_index = fully_connected_edge_index(num_hits, device=node_features.device)
        edge_attr = build_edge_attr(node_features, edge_index)

        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        data.u = torch.tensor([[energy.sum()]], dtype=torch.float)

        # Target angle vector (3 dims)
        if item.angle is not None:
            angle_vec = np.asarray(item.angle, dtype=np.float32).reshape(-1)
        elif item.true_angle_vector is not None:
            angle_vec = np.asarray(item.true_angle_vector, dtype=np.float32).reshape(-1)
        else:
            angle_vec = np.zeros(3, dtype=np.float32)
        if angle_vec.size < 3:
            angle_vec = np.pad(angle_vec, (0, 3 - angle_vec.size), mode="constant")
        data.y = torch.tensor(angle_vec[:3], dtype=torch.float).unsqueeze(0)  # [1,3] per graph

        if item.event_id is not None:
            data.event_id = torch.tensor(int(item.event_id), dtype=torch.long)
        if item.group_id is not None:
            data.group_id = torch.tensor(int(item.group_id), dtype=torch.long)
        if item.pion_stop is not None:
            data.pion_stop = torch.tensor(item.pion_stop, dtype=torch.float).unsqueeze(0)

        return data

    @staticmethod
    def _coerce(raw: Dict[str, Any] | PositronAngleRecord) -> PositronAngleRecord:
        if isinstance(raw, PositronAngleRecord):
            return raw
        if hasattr(raw, "coord"):
            return PositronAngleRecord(
                coord=getattr(raw, "coord"),
                z=getattr(raw, "z"),
                energy=getattr(raw, "energy"),
                view=getattr(raw, "view"),
                angle=getattr(raw, "angle", None),
                event_id=getattr(raw, "event_id", None),
                group_id=getattr(raw, "group_id", None),
                pion_stop=getattr(raw, "pion_stop", None) or getattr(raw, "true_pion_stop", None),
                true_angle_vector=getattr(raw, "true_angle_vector", None),
            )
        return PositronAngleRecord(
            coord=raw["coord"],
            z=raw["z"],
            energy=raw["energy"],
            view=raw["view"],
            angle=raw.get("angle", [0.0, 0.0]),
            event_id=raw.get("event_id"),
            group_id=raw.get("group_id"),
            pion_stop=raw.get("pion_stop") or raw.get("true_pion_stop"),
            true_angle_vector=raw.get("true_angle_vector"),
        )
