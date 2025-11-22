"""
Backwards-compatible entry point for model classes.

New code should import from the dedicated classifier/regressor modules:
    - pioneerml.models.classifiers
    - pioneerml.models.regressors
    - pioneerml.models.blocks
"""

from pioneerml.models.blocks import FullGraphTransformerBlock
from pioneerml.models.classifiers import GroupClassifier, GroupSplitter, GroupAffinityModel
from pioneerml.models.regressors import PionStopRegressor, EndpointRegressor, PositronAngleModel

__all__ = [
    "FullGraphTransformerBlock",
    "GroupClassifier",
    "GroupSplitter",
    "GroupAffinityModel",
    "PionStopRegressor",
    "EndpointRegressor",
    "PositronAngleModel",
]
