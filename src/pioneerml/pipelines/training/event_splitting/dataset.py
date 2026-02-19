from dataclasses import dataclass
from typing import Any

from pioneerml.common.loader import GraphTrainingDataset


@dataclass
class EventSplitterDataset(GraphTrainingDataset):
    """Dataset wrapper for event-splitter training."""

    loader_factory: Any | None = None
    loader: Any | None = None
