"""
GNN model architectures for PIONEER reconstruction tasks.

Standard graph representation:
- Node features (4D): [coord, z, energy, view]
- Edge features (4D): [dx, dz, dE, same_view]
- Global feature: data.u = total group energy
"""

from pioneerml.common.models.blocks import FullGraphTransformerBlock
from pioneerml.common.models.classifiers import (
    GroupClassifier,
    GroupClassifierStereo,
    GroupSplitter,
    GroupAffinityModel,
)
from pioneerml.common.models.regressors import (
    PionStopRegressor,
    EndpointRegressor,
    OrthogonalEndpointRegressor,
    PositronAngleModel,
)
from pioneerml.common.models.event_builder import EventBuilder

__all__ = [
    "FullGraphTransformerBlock",
    "GroupClassifier",
    "GroupClassifierStereo",
    "GroupSplitter",
    "GroupAffinityModel",
    "PionStopRegressor",
    "EndpointRegressor",
    "OrthogonalEndpointRegressor",
    "PositronAngleModel",
    "EventBuilder",
]
