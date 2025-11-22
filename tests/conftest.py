"""
Pytest configuration and shared fixtures.
"""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data


@pytest.fixture
def device():
    """Get available device (CPU or CUDA)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_graph_data():
    """Create a sample graph data object for testing."""
    num_nodes = 10

    # Node features: [coord, z, energy, view, group_energy]
    x = torch.randn(num_nodes, 5)

    # Fully connected edges (no self-loops)
    edge_index = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()

    # Edge features: [dx, dz, dE, same_view]
    num_edges = edge_index.shape[1]
    edge_attr = torch.randn(num_edges, 4)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.batch = torch.zeros(num_nodes, dtype=torch.long)  # single graph batch assignment
    return data


@pytest.fixture
def sample_graph_record():
    """Create a sample GraphRecord for dataset testing."""
    num_hits = 15
    return {
        "coord": np.random.randn(num_hits).astype(np.float32),
        "z": np.random.randn(num_hits).astype(np.float32),
        "energy": np.abs(np.random.randn(num_hits)).astype(np.float32),
        "view": np.random.randint(0, 2, num_hits).astype(np.float32),
        "labels": [0, 1],  # Contains pion and muon
        "event_id": 42,
        "group_id": 7,
    }


@pytest.fixture
def sample_pion_stop_record():
    """Create a sample PionStopRecord for testing."""
    num_hits = 20
    return {
        "coord": np.random.randn(num_hits).astype(np.float32),
        "z": np.random.randn(num_hits).astype(np.float32),
        "energy": np.abs(np.random.randn(num_hits)).astype(np.float32),
        "view": np.random.randint(0, 2, num_hits).astype(np.float32),
        "time": np.sort(np.random.rand(num_hits)).astype(np.float32),
        "pdg": np.random.choice([1, 2, 3], num_hits).astype(np.int32),
        "true_x": np.random.randn(num_hits).astype(np.float32),
        "true_y": np.random.randn(num_hits).astype(np.float32),
        "true_z": np.random.randn(num_hits).astype(np.float32),
        "true_time": np.sort(np.random.rand(num_hits)).astype(np.float32),
        "event_id": 42,
        "group_id": 7,
    }
