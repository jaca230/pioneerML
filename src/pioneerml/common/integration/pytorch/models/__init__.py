"""
GNN model architectures for PIONEER reconstruction tasks.

Standard graph representation:
- Node features (4D): [coord, z, energy, view]
- Edge features (4D): [dx, dz, dE, same_view]
- Optional graph-level features are model-specific (for example `group_probs`, `u`).
"""

from pioneerml.common.integration.pytorch.models.architectures import (
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
    list_registered_architectures,
    register_architecture,
    resolve_architecture,
)
from pioneerml.common.integration.pytorch.models.primitives import (
    FullGraphTransformerBlock,
    QuantileOutputHead,
    ViewAwareEncoder,
)
from pioneerml.common.integration.pytorch.losses import AngularUnitVectorLoss, QuantileAngularLoss, QuantilePinballLoss

__all__ = [
    "register_architecture",
    "resolve_architecture",
    "list_registered_architectures",
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
    "AngularUnitVectorLoss",
    "QuantileAngularLoss",
    "QuantilePinballLoss",
]
