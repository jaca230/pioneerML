from dataclasses import dataclass
from typing import Any

from pioneerml.common.pipeline_utils.loader import GraphTrainingDataset


@dataclass
class GroupSplitterDataset(GraphTrainingDataset):
    """Dataset wrapper for the group splitting pipeline."""

    loader: Any | None = None
