"""EndpointGraphDataset for endpoint regression."""

from __future__ import annotations

from typing import Sequence, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from pioneerml.data.datasets.graph_group import GraphRecord
from pioneerml.data.datasets.utils import fully_connected_edge_index, build_edge_attr


class EndpointGraphDataset(Dataset):
    """
    Dataset for endpoint regression.

    Produces graph Data with node features [coord, z, energy, view], edge features,
    and target endpoints repeated across quantiles: shape [2, 3, num_quantiles].
    """

    def __init__(self, records: Sequence[GraphRecord | Dict[str, Any]], *, num_quantiles: int = 3):
        self.items = [self._coerce(item) for item in records]
        self.num_quantiles = num_quantiles

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

        if item.true_start is None or item.true_end is None:
            raise ValueError("EndpointGraphDataset requires true_start and true_end for each record.")

        endpoints = torch.tensor(
            np.stack([item.true_start, item.true_end], axis=0), dtype=torch.float
        )  # [2, 3]
        target = endpoints.unsqueeze(-1).repeat(1, 1, self.num_quantiles)  # [2, 3, Q]
        # Add graph dimension so batching matches model output [batch, 2, 3, Q]
        data.y = target.unsqueeze(0)

        if item.event_id is not None:
            data.event_id = torch.tensor(int(item.event_id), dtype=torch.long)
        if item.group_id is not None:
            data.group_id = torch.tensor(int(item.group_id), dtype=torch.long)

        if item.group_probs is not None:
            data.group_probs = torch.tensor(item.group_probs, dtype=torch.float).unsqueeze(0)

        return data

    @staticmethod
    def _coerce(raw: GraphRecord | Dict[str, Any]) -> GraphRecord:
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
            true_start=raw.get("true_start"),
            true_end=raw.get("true_end"),
            true_pion_stop=raw.get("true_pion_stop"),
            true_angle_vector=raw.get("true_angle_vector"),
            pred_pion_stop=raw.get("pred_pion_stop"),
            matched_pion_index=raw.get("matched_pion_index"),
            pion_stop_for_angle=raw.get("pion_stop_for_angle"),
            true_arc_length=raw.get("true_arc_length"),
            pred_endpoints=raw.get("pred_endpoints"),
        )
