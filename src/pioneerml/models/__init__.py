"""
GNN model architectures for PIONEER reconstruction tasks.

All models use a standardized graph representation:
- Node features (5D): [coord, z, energy, view, group_energy]
- Edge features (4D): [dx, dz, dE, same_view]
- Architecture: Transformer-based with JumpingKnowledge and attentional pooling
"""

from pioneerml.models.blocks import FullGraphTransformerBlock
from pioneerml.models.classifiers import (
    GroupClassifier,
    GroupSplitter,
    GroupAffinityModel,
)
from pioneerml.models.regressors import (
    PionStopRegressor,
    EndpointRegressor,
    PositronAngleModel,
)

__all__ = [
    "FullGraphTransformerBlock",
    "GroupClassifier",
    "GroupSplitter",
    "GroupAffinityModel",
    "PionStopRegressor",
    "EndpointRegressor",
    "PositronAngleModel",
]
