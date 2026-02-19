from dataclasses import dataclass
from typing import Any

from pioneerml.common.loader import GraphTrainingDataset


@dataclass
class EndpointRegressorDataset(GraphTrainingDataset):
    """Dataset wrapper for time-group endpoint-regressor training."""

    loader_factory: Any | None = None
    loader: Any | None = None
