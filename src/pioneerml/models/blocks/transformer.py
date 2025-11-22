"""
Transformer block for graph neural networks.

This module contains the core transformer building block used across
all PIONEER ML model architectures.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv


class FullGraphTransformerBlock(nn.Module):
    """
    Graph Transformer block with pre-normalization and residual connections.

    This is the fundamental building block for all PIONEER ML models. It combines:
    - Multi-head graph attention (TransformerConv)
    - Feed-forward network
    - Layer normalization (pre-norm style)
    - Residual connections

    The block processes node features while attending to graph structure
    and edge attributes.

    Args:
        hidden: Hidden dimension size.
        heads: Number of attention heads (default: 4).
        edge_dim: Edge feature dimension (default: 4).
        dropout: Dropout rate (default: 0.1).

    Example:
        >>> block = FullGraphTransformerBlock(hidden=128, heads=4)
        >>> x = torch.randn(10, 128)  # 10 nodes
        >>> edge_index = torch.randint(0, 10, (2, 40))
        >>> edge_attr = torch.randn(40, 4)
        >>> out = block(x, edge_index, edge_attr)
        >>> assert out.shape == (10, 128)
    """

    def __init__(
        self,
        hidden: int,
        heads: int = 4,
        edge_dim: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Pre-normalization is more stable for deep transformers
        self.ln1 = nn.LayerNorm(hidden)

        # Multi-head graph attention
        self.attn = TransformerConv(
            in_channels=hidden,
            out_channels=hidden // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
            concat=True,  # Concatenate heads
            beta=True,    # Use attention scaling
        )

        # Feed-forward network with expansion
        self.ln2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, 4 * hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden, hidden),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the transformer block.

        Args:
            x: Node features [num_nodes, hidden].
            edge_index: Graph connectivity [2, num_edges].
            edge_attr: Edge features [num_edges, edge_dim].

        Returns:
            Updated node features [num_nodes, hidden].
        """
        # Multi-head graph attention with residual
        h = self.attn(self.ln1(x), edge_index, edge_attr)
        x = x + h

        # Feed-forward network with residual
        h2 = self.ffn(self.ln2(x))
        x = x + h2

        return x
