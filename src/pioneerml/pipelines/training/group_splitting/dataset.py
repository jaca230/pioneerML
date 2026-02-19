from dataclasses import dataclass
from typing import Any

from pioneerml.common.loader import GraphTrainingDataset


@dataclass
class GroupSplitterDataset(GraphTrainingDataset):
    """Dataset wrapper for the group splitting pipeline."""

    loader_factory: Any | None = None
    loader: Any | None = None
