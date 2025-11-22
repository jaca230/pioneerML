"""
Tests for GNN model classes.
"""

import pytest
import torch
from pioneerml.models import (
    FullGraphTransformerBlock,
    GroupClassifier,
    GroupAffinityModel,
    GroupSplitter,
    PionStopRegressor,
    EndpointRegressor,
    PositronAngleModel,
)


class TestFullGraphTransformerBlock:
    """Tests for the FullGraphTransformerBlock."""

    def test_forward_pass(self, sample_graph_data, device):
        """Test that forward pass works and preserves dimensions."""
        block = FullGraphTransformerBlock(hidden=64, heads=4, edge_dim=4, dropout=0.1)
        block = block.to(device)
        data = sample_graph_data.to(device)

        # Project input to hidden dim
        x = torch.randn(data.x.size(0), 64, device=device)

        output = block(x, data.edge_index, data.edge_attr)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_different_heads(self, sample_graph_data, device):
        """Test with different numbers of attention heads."""
        data = sample_graph_data.to(device)
        x = torch.randn(data.x.size(0), 128, device=device)

        for heads in [1, 2, 4, 8]:
            block = FullGraphTransformerBlock(hidden=128, heads=heads, edge_dim=4)
            block = block.to(device)
            output = block(x, data.edge_index, data.edge_attr)
            assert output.shape == x.shape


class TestGroupClassifier:
    """Tests for GroupClassifier model."""

    def test_output_shape(self, sample_graph_data, device):
        """Test that output shape is correct."""
        model = GroupClassifier(in_dim=5, hidden=64, num_blocks=2, num_classes=3)
        model = model.to(device)
        data = sample_graph_data.to(device)

        output = model(data)

        assert output.shape == (1, 3)  # Single graph, 3 classes
        assert not torch.isnan(output).any()

    def test_different_configs(self, sample_graph_data, device):
        """Test with different architecture configurations."""
        data = sample_graph_data.to(device)

        configs = [
            {"hidden": 128, "num_blocks": 1, "heads": 2},
            {"hidden": 200, "num_blocks": 2, "heads": 4},
            {"hidden": 256, "num_blocks": 3, "heads": 8},
        ]

        for config in configs:
            model = GroupClassifier(**config, num_classes=3)
            model = model.to(device)
            output = model(data)
            assert output.shape == (1, 3)


class TestGroupSplitter:
    """Tests for GroupSplitter model."""

    def test_output_shape(self, sample_graph_data, device):
        """Test that output is per-node predictions."""
        model = GroupSplitter(in_channels=5, hidden=128, layers=3, num_classes=3)
        model = model.to(device)
        data = sample_graph_data.to(device)

        output = model(data)

        assert output.shape == (data.x.size(0), 3)  # Per node, 3 classes
        assert not torch.isnan(output).any()


class TestPionStopRegressor:
    """Tests for PionStopRegressor model."""

    def test_output_shape(self, sample_graph_data, device):
        """Test that output is 3D coordinate."""
        model = PionStopRegressor(in_channels=5, hidden=128, layers=3)
        model = model.to(device)
        data = sample_graph_data.to(device)

        output = model(data)

        assert output.shape == (1, 3)  # Single graph, 3D coordinate
        assert not torch.isnan(output).any()


class TestEndpointRegressor:
    """Tests for EndpointRegressor model."""

    def test_output_shape(self, sample_graph_data, device):
        """Test that output is two 3D coordinates."""
        model = EndpointRegressor(in_channels=5, hidden=160, layers=2)
        model = model.to(device)
        data = sample_graph_data.to(device)

        output = model(data)

        assert output.shape == (1, 2, 3)  # Single graph, 2 points, 3D
        assert not torch.isnan(output).any()


class TestPositronAngleModel:
    """Tests for PositronAngleModel."""

    def test_output_shape(self, sample_graph_data, device):
        """Test that output is 2 angle components."""
        model = PositronAngleModel(in_channels=5, hidden=128, layers=2)
        model = model.to(device)
        data = sample_graph_data.to(device)

        output = model(data)

        assert output.shape == (1, 2)  # Single graph, 2 angles
        assert not torch.isnan(output).any()


class TestGroupAffinityModel:
    """Tests for GroupAffinityModel."""

    def test_output_shape(self, sample_graph_data, device):
        """Test that output is single affinity score."""
        model = GroupAffinityModel(in_channels=5, hidden_channels=128, num_layers=3)
        model = model.to(device)
        data = sample_graph_data.to(device)

        output = model(data)

        assert output.shape == (1, 1)  # Single graph, single score
        assert not torch.isnan(output).any()
