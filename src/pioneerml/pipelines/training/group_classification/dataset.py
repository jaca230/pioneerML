from dataclasses import dataclass
from typing import Any

from pioneerml.common.loader import GraphTrainingDataset


@dataclass
class GroupClassifierDataset(GraphTrainingDataset):
    """Dataset wrapper for the group classification pipeline."""

    loader_factory: Any | None = None
    loader: Any | None = None
