from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import Data


@dataclass
class GraphTrainingDataset:
    """Generic dataset wrapper used by graph training pipelines."""

    data: Data
    targets: torch.Tensor
