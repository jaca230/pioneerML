"""
Regression models for PIONEER ML.
"""

from pioneerml.common.models.regressors.endpoint_regressor import EndpointRegressor, OrthogonalEndpointRegressor
from pioneerml.common.models.regressors.pion_stop import PionStopRegressor
from pioneerml.common.models.regressors.positron_angle import PositronAngleModel

PionStop = PionStopRegressor
PositronAngle = PositronAngleModel

__all__ = [
    "PionStopRegressor",
    "PionStop",
    "EndpointRegressor",
    "OrthogonalEndpointRegressor",
    "PositronAngleModel",
    "PositronAngle",
]
