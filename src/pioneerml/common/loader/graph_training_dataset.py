from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GraphTrainingDataset:
    data: Any
    targets: Any
