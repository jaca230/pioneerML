"""
PionStopGraphDataset and supporting record dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Optional, Dict, Any, List

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from pioneerml.data.datasets.utils import fully_connected_edge_index, build_edge_attr


@dataclass
class PionStopRecord:
    coord: Iterable[float]
    z: Iterable[float]
    energy: Iterable[float]
    view: Iterable[float]
    time: Iterable[float] | None = None
    pdg: Iterable[int] | None = None
    true_x: Iterable[float] | None = None
    true_y: Iterable[float] | None = None
    true_z: Iterable[float] | None = None
    true_time: Iterable[float] | None = None
    true_pion_stop: Optional[Sequence[float]] = None
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

        # If a pre-computed true pion stop is provided, use it directly.
        if item.true_pion_stop is not None:
            stop_target = np.asarray(item.true_pion_stop, dtype=np.float32)
            if stop_target.shape != (3,):
                stop_target = stop_target.reshape(-1)[:3]
        else:
            pdg = np.asarray(item.pdg, dtype=np.int32)
            true_x = np.asarray(item.true_x, dtype=np.float32)
            true_y = np.asarray(item.true_y, dtype=np.float32)
            true_z = np.asarray(item.true_z, dtype=np.float32)
            true_time = np.asarray(item.true_time, dtype=np.float32)
            hit_time = np.asarray(item.time, dtype=np.float32)

            if not (
                pdg.shape
                == true_x.shape
                == true_y.shape
                == true_z.shape
                == true_time.shape
                == hit_time.shape
                == coord.shape
            ):
                raise ValueError("All PionStopRecord arrays must align per hit.")

            pion_indices = np.flatnonzero(pdg == self.pion_pdg)
            if pion_indices.size < self.min_pion_hits:
                raise ValueError("Record does not contain enough pion hits to compute stop target.")

            ref_time = true_time[pion_indices] if self.use_true_time else hit_time[pion_indices]
            last_idx = pion_indices[int(np.argmax(ref_time))]
            stop_target = np.array([true_x[last_idx], true_y[last_idx], true_z[last_idx]], dtype=np.float32)

        num_hits = coord.shape[0]
        node_features = torch.tensor(np.stack([coord, z_pos, energy, view], axis=1), dtype=torch.float)

        edge_index = fully_connected_edge_index(num_hits, device=node_features.device)
        edge_attr = build_edge_attr(node_features, edge_index)

        target_tensor = torch.tensor(stop_target, dtype=torch.float).unsqueeze(0)

        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=target_tensor,
        )
        data.u = torch.tensor([[energy.sum()]], dtype=torch.float)

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
            time=raw.get("time"),
            pdg=raw.get("pdg"),
            true_x=raw.get("true_x"),
            true_y=raw.get("true_y"),
            true_z=raw.get("true_z"),
            true_time=raw.get("true_time"),
            true_pion_stop=raw.get("true_pion_stop"),
            event_id=raw.get("event_id"),
            group_id=raw.get("group_id"),
        )
