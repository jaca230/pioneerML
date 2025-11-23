"""
SplitterGraphDataset for per-hit classification.
"""

from __future__ import annotations

from typing import Sequence, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from pioneerml.data.datasets.graph_group import GraphRecord
from pioneerml.data.datasets.utils import fully_connected_edge_index, build_edge_attr


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
