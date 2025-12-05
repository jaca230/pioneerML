"""
Group classifier model for multi-label time group classification.

Identifies which particle types are present in a time-grouped hit collection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import JumpingKnowledge, AttentionalAggregation
from torch_geometric.data import Data

from pioneerml.models.base import GraphModel
from pioneerml.models.blocks import FullGraphTransformerBlock


class GroupClassifier(GraphModel):
    """
    Multi-label classifier for time group particle identification.

    Classifies time groups as containing pions, muons, and/or MIPs (minimum
    ionizing particles like positrons/electrons).

    Architecture:
    - Input embedding
    - N transformer blocks with residual connections
    - Jumping Knowledge aggregation (concatenates all layers)
    - Attentional pooling (graph-level)
    - Classification head

    Args:
        in_dim: Input node feature dimension (default: 5).
        edge_dim: Edge feature dimension (default: 4).
        hidden: Hidden dimension size (default: 200).
        heads: Number of attention heads (default: 4).
        num_blocks: Number of transformer blocks (default: 2).
        dropout: Dropout rate (default: 0.1).
        num_classes: Number of output classes (default: 3 for pi/mu/mip).

    Example:
        >>> model = GroupClassifier(hidden=200, num_blocks=2, num_classes=3)
        >>> predictions = model(graph_data)  # [batch_size, 3]
    """

    def __init__(
        self,
        in_dim: int = 5,
        edge_dim: int = 4,
        hidden: int = 200,
        heads: int = 4,
        num_blocks: int = 2,
        dropout: float = 0.1,
        num_classes: int = 3,
    ):
        super().__init__(in_channels=in_dim, hidden=hidden, edge_dim=edge_dim, dropout=dropout)

        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.heads = heads

        # Input projection
        self.input_embed = nn.Linear(in_dim, hidden)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                FullGraphTransformerBlock(
                    hidden=hidden,
                    heads=heads,
                    edge_dim=edge_dim,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        # Jumping Knowledge (concatenate all layers)
        self.jk = JumpingKnowledge(mode="cat")
        concat_dim = hidden * num_blocks

        # Attentional aggregation for graph-level pooling
        self.pool = AttentionalAggregation(
            nn.Sequential(
                nn.Linear(concat_dim, concat_dim // 2),
                nn.ReLU(),
                nn.Linear(concat_dim // 2, 1),
            )
        )

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(concat_dim, concat_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(concat_dim // 2, num_classes),
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass.

        Args:
            data: PyG Data object with x, edge_index, edge_attr, batch.

        Returns:
            Class logits [batch_size, num_classes].
        """
        pooled = self.extract_embeddings(data)
        return self.head(pooled)

    def extract_embeddings(self, data: Data) -> torch.Tensor:
        """Return graph-level embeddings before the classification head."""
        x = self.input_embed(data.x)
        xs = []
        for block in self.blocks:
            x = block(x, data.edge_index, data.edge_attr)
            xs.append(x)
        x_cat = self.jk(xs)
        return self.pool(x_cat, data.batch)

    def summary(self) -> dict:
        """Get model summary."""
        info = super().summary()
        info.update(
            {
                "num_blocks": self.num_blocks,
                "num_classes": self.num_classes,
                "heads": self.heads,
            }
        )
        return info
