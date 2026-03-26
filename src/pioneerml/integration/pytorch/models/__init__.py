"""
GNN model architectures for PIONEER reconstruction tasks.

Standard graph representation:
- Node features (4D): [coord, z, energy, view]
- Edge features (4D): [dx, dz, dE, same_view]
- Optional graph-level features are model-specific (for example `group_probs`, `u`).
"""

from pioneerml.integration.pytorch.models.architectures import (
    ARCHITECTURE_REGISTRY,
    ArchitectureFactory,
)
from pioneerml.integration.pytorch.models.primitives import (
    FullGraphTransformerBlock,
    QuantileOutputHead,
    ViewAwareEncoder,
)

__all__ = [
    "ARCHITECTURE_REGISTRY",
    "ArchitectureFactory",
    "FullGraphTransformerBlock",
    "ViewAwareEncoder",
    "QuantileOutputHead",
]
