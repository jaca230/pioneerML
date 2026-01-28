"""
Tests for dataset classes and graph utilities.
"""

import pytest
import torch
import numpy as np
from pioneerml.data import (
    GraphRecord,
    PionStopRecord,
    GraphGroupDataset,
    PionStopGraphDataset,
    SplitterGraphDataset,
    fully_connected_edge_index,
    build_edge_attr,
)


class TestGraphUtilities:
    """Tests for graph construction utilities."""

    def test_fully_connected_edge_index(self):
        """Test fully connected edge index generation."""
        # 3 nodes
        edge_index = fully_connected_edge_index(3)
        assert edge_index.shape == (2, 6)  # 3 * 2 = 6 edges (no self-loops)

        # Verify no self-loops
        src, dst = edge_index
        assert not (src == dst).any()

        # 5 nodes
        edge_index = fully_connected_edge_index(5)
        assert edge_index.shape == (2, 20)  # 5 * 4 = 20 edges

    def test_fully_connected_single_node(self):
        """Test edge case: single node (no edges)."""
        edge_index = fully_connected_edge_index(1)
        assert edge_index.shape == (2, 0)

    def test_build_edge_attr(self):
        """Test edge attribute computation."""
        # Create simple node features
        node_features = torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0, 5.0],  # coord, z, energy, view, group_energy
                [1.0, 1.0, 2.0, 0.0, 5.0],
                [2.0, 2.0, 3.0, 1.0, 5.0],
            ]
        )

        edge_index = fully_connected_edge_index(3)
        edge_attr = build_edge_attr(node_features, edge_index)

        assert edge_attr.shape == (6, 4)  # 6 edges, 4 attributes each

        # Check specific edge (0 -> 1)
        src_to_dst_idx = 0  # First edge
        dx, dz, dE, same_view = edge_attr[src_to_dst_idx]
        assert dx == pytest.approx(1.0)  # 1.0 - 0.0
        assert dz == pytest.approx(1.0)  # 1.0 - 0.0
        assert dE == pytest.approx(1.0)  # 2.0 - 1.0
        assert same_view == 1.0  # Both have view=0


class TestGraphGroupDataset:
    """Tests for GraphGroupDataset."""

    def test_dataset_length(self, sample_graph_record):
        """Test dataset length."""
        records = [sample_graph_record for _ in range(10)]
        dataset = GraphGroupDataset(records, num_classes=3)
        assert len(dataset) == 10

    def test_getitem(self, sample_graph_record):
        """Test retrieving a single item."""
        dataset = GraphGroupDataset([sample_graph_record], num_classes=3)
        data = dataset[0]

        # Check data structure
        assert hasattr(data, "x")
        assert hasattr(data, "edge_index")
        assert hasattr(data, "edge_attr")
        assert hasattr(data, "y")

        # Check dimensions
        num_hits = len(sample_graph_record["coord"])
        assert data.x.shape == (num_hits, 4)  # stereo coord, z, energy, view
        assert data.edge_attr.shape[1] == 4  # 4D edge features
        assert data.y.shape == (3,)  # 3 classes
        assert data.hit_mask.sum() == num_hits

    def test_labels(self, sample_graph_record):
        """Test multi-label encoding."""
        dataset = GraphGroupDataset([sample_graph_record], num_classes=3)
        data = dataset[0]

        # Record has labels [0, 1], so y should be [1, 1, 0]
        expected = torch.tensor([1.0, 1.0, 0.0])
        assert torch.allclose(data.y, expected)

    def test_graph_record_dataclass(self):
        """Test using GraphRecord dataclass directly."""
        record = GraphRecord(
            coord=[1.0, 2.0, 3.0],
            z=[0.0, 1.0, 2.0],
            energy=[1.5, 2.0, 1.0],
            view=[0.0, 1.0, 0.0],
            hit_mask=[True, True, False],
            time_group_ids=[0, 0, -1],
            labels=[0],
            event_id=123,
            group_id=5,
        )

        dataset = GraphGroupDataset([record], num_classes=3)
        data = dataset[0]

        assert data.event_id == 123
        assert data.group_id == 5
        assert data.num_valid_hits.item() == 2
        assert data.edge_index.size(1) == 2 * 1  # 2 nodes fully connected (directed)
        assert torch.equal(data.time_group_ids, torch.tensor([0, 0, -1]))


class TestPionStopGraphDataset:
    """Tests for PionStopGraphDataset."""

    def test_dataset_creation(self, sample_pion_stop_record):
        """Test creating dataset with pion stop records."""
        # Ensure at least one pion hit
        sample_pion_stop_record["pdg"][0] = 1  # Set first hit to pion

        dataset = PionStopGraphDataset([sample_pion_stop_record], pion_pdg=1, min_pion_hits=1)
        assert len(dataset) == 1

    def test_stop_target_extraction(self, sample_pion_stop_record):
        """Test that pion stop target is correctly extracted."""
        # Set last pion hit
        sample_pion_stop_record["pdg"][-1] = 1
        sample_pion_stop_record["true_time"][-1] = 100.0  # Latest time

        dataset = PionStopGraphDataset([sample_pion_stop_record], pion_pdg=1)
        data = dataset[0]

        # Target should be the true position of the last pion hit
        expected_target = torch.tensor(
            [
                [
                    sample_pion_stop_record["true_x"][-1],
                    sample_pion_stop_record["true_y"][-1],
                    sample_pion_stop_record["true_z"][-1],
                ]
            ]
        )
        assert torch.allclose(data.y, expected_target)

    def test_insufficient_pion_hits(self, sample_pion_stop_record):
        """Test error when insufficient pion hits."""
        # Set all PDG codes to non-pion
        sample_pion_stop_record["pdg"][:] = 2  # All muons

        dataset = PionStopGraphDataset([sample_pion_stop_record], pion_pdg=1, min_pion_hits=1)

        with pytest.raises(ValueError, match="does not contain enough pion hits"):
            _ = dataset[0]


class TestSplitterGraphDataset:
    """Tests for SplitterGraphDataset."""

    def test_dataset_with_hit_labels(self, sample_graph_record):
        """Test dataset with per-hit labels."""
        num_hits = len(sample_graph_record["coord"])
        sample_graph_record["hit_labels"] = np.random.randint(0, 2, (num_hits, 3)).astype(np.float32)

        dataset = SplitterGraphDataset([sample_graph_record], use_group_probs=False)
        data = dataset[0]

        assert data.x.shape == (num_hits, 4)  # Base stereo features
        assert data.y.shape == (num_hits, 3)  # Per-hit 3-class labels

    def test_dataset_with_group_probs(self, sample_graph_record):
        """Test dataset with group classifier probabilities."""
        num_hits = len(sample_graph_record["coord"])
        sample_graph_record["hit_labels"] = np.random.randint(0, 2, (num_hits, 3)).astype(np.float32)
        sample_graph_record["group_probs"] = [0.8, 0.3, 0.1]  # [p_pi, p_mu, p_mip]

        dataset = SplitterGraphDataset([sample_graph_record], use_group_probs=True)
        data = dataset[0]

        assert data.x.shape == (num_hits, 4)  # features remain separate from probs

    def test_missing_hit_labels_error(self, sample_graph_record):
        """Test error when hit_labels are missing."""
        # Don't include hit_labels
        dataset = SplitterGraphDataset([sample_graph_record])

        with pytest.raises(ValueError, match="requires GraphRecord.hit_labels"):
            _ = dataset[0]
