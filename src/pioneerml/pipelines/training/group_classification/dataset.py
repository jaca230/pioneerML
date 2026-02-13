from dataclasses import dataclass
from typing import Any

from pioneerml.common.pipeline_utils.loader import GraphTrainingDataset


@dataclass
class GroupClassifierDataset(GraphTrainingDataset):
    """Dataset wrapper for the group classification pipeline."""

    loader: Any | None = None
