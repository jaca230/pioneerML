"""
Utility functions for building graph edges and attributes.
"""

from __future__ import annotations

from typing import Optional

import torch


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
