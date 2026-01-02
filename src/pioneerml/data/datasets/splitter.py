"""SplitterGraphDataset for per-hit classification."""

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

    Uses standardized node features [coord, z, energy, view] and optional
    group probabilities stored separately on the Data object.
    Expects per-hit multi-label targets in GraphRecord.hit_labels with shape
    [num_hits, 3] corresponding to [is_pion, is_muon, is_mip].
    """

    def __init__(
        self,
        records: Sequence[GraphRecord | dict],
        *,
        use_group_probs: bool = False,
    ):
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

        node_features = np.stack([coord, z_pos, energy, view], axis=1)  # [N, 4]
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
        data.u = torch.tensor([[energy.sum()]], dtype=torch.float)

        if item.group_probs is not None and self.use_group_probs:
            data.group_probs = torch.tensor(item.group_probs, dtype=torch.float).unsqueeze(0)

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
            hit_pdgs=raw.get("hit_pdgs"),
            class_energies=raw.get("class_energies"),
            true_pion_stop=raw.get("true_pion_stop"),
            true_angle_vector=raw.get("true_angle_vector"),
            pred_pion_stop=raw.get("pred_pion_stop"),
            matched_pion_index=raw.get("matched_pion_index"),
            pion_stop_for_angle=raw.get("pion_stop_for_angle"),
        )
