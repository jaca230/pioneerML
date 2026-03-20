"""
GNN model architectures for PIONEER reconstruction tasks.

Standard graph representation:
- Node features (4D): [coord, z, energy, view]
- Edge features (4D): [dx, dz, dE, same_view]
- Optional graph-level features are model-specific (for example `group_probs`, `u`).
"""

from pioneerml.common.integration.pytorch.models.architectures import (
    ARCHITECTURE_REGISTRY,
    ArchitectureFactory,
    EventSplitter,
    GroupClassifier,
    GroupClassifierStereo,
    GroupSplitter,
    GroupAffinityModel,
    PionStopRegressor,
    EndpointRegressor,
    OrthogonalEndpointRegressor,
    PositronAngleModel,
)
from pioneerml.common.integration.pytorch.models.primitives import (
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
    "GroupClassifier",
    "GroupClassifierStereo",
    "GroupSplitter",
    "GroupAffinityModel",
    "PionStopRegressor",
    "EndpointRegressor",
    "OrthogonalEndpointRegressor",
    "PositronAngleModel",
    "EventSplitter",
]
