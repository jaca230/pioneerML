"""
Utility functions for building graph edges and attributes.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Any

import numpy as np

import torch
from torch_geometric.utils import dense_to_sparse


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


def build_event_graph(container: Any, device: torch.device, radius_z: float = 0.5) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor
]:
    """
    Converts an EventContainer to graph inputs for the EventBuilder.
    Builds inter-group connections based on Z-distance only (stereo-aware).

    Returns:
        x: Node features [TotalHits, 25]
        edge_index: Edge list
        edge_attr: Edge features
        group_indices: Per-hit group assignment
        num_groups: Number of groups in the event
        affinity_targets: [num_groups, num_groups] same-origin mask
    """
    node_features_list = []
    group_indices_list = []
    group_origins = []

    for g_idx, record in enumerate(container.records):
        coords = record.coord
        zs = record.z
        energies = record.energy
        views = record.view
        num_hits = len(coords)

        # Broadcast group-level probabilities [3] -> [num_hits, 3]
        if record.group_probs is not None:
            probs = torch.tensor(record.group_probs, dtype=torch.float32).repeat(num_hits, 1)
        else:
            probs = torch.zeros(num_hits, 3)

        # Broadcast predicted endpoints [18] -> [num_hits, 18]
        if record.pred_endpoints is not None:
            eps = torch.tensor(record.pred_endpoints, dtype=torch.float32).view(-1).repeat(num_hits, 1)
        else:
            eps = torch.zeros(num_hits, 18)

        base = torch.tensor(np.stack([coords, zs, energies, views], axis=1), dtype=torch.float32)

        # Concatenate Features: [4] + [3] + [18] = 25
        feats = torch.cat([base, probs, eps], dim=1)
        node_features_list.append(feats)

        group_indices_list.append(torch.full((num_hits,), g_idx, dtype=torch.long))
        group_origins.append(container.origins[g_idx])

    if not node_features_list:
        raise ValueError("No groups provided to build_event_graph.")

    x = torch.cat(node_features_list, dim=0).to(device)  # [TotalHits, 25]
    group_indices = torch.cat(group_indices_list, dim=0).to(device)  # [TotalHits]
    num_groups = len(container.records)

    origins = torch.tensor(group_origins, device=device)
    affinity_targets = (origins.unsqueeze(1) == origins.unsqueeze(0)).float()

    # Build edges: fully connected within a group, z-radius across groups
    z_col = x[:, 1]
    g_i = group_indices.unsqueeze(1)
    g_j = group_indices.unsqueeze(0)

    intra_mask = g_i == g_j

    dist_z = torch.abs(z_col.unsqueeze(1) - z_col.unsqueeze(0))
    inter_mask = (dist_z < radius_z) & (g_i != g_j)

    final_adj = intra_mask | inter_mask
    edge_index, _ = dense_to_sparse(final_adj)

    src, dst = edge_index
    u, v = x[src], x[dst]

    diffs = u[:, :3] - v[:, :3]  # coord, z, energy diffs
    is_same_view = (u[:, 3] == v[:, 3]).float().unsqueeze(1)
    is_same_group = (group_indices[src] == group_indices[dst]).float().unsqueeze(1)

    edge_attr = torch.cat([diffs, is_same_view, is_same_group], dim=1)

    return x, edge_index, edge_attr, group_indices, num_groups, affinity_targets
