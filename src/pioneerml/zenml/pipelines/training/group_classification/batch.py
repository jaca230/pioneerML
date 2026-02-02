from dataclasses import dataclass

import torch
from torch_geometric.data import Data


@dataclass
class GroupClassifierBatch:
    data: Data
    targets: torch.Tensor
    target_energy: torch.Tensor
