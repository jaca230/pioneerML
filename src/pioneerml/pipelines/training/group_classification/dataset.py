from dataclasses import dataclass

import torch
from torch_geometric.data import Data


@dataclass
class GroupClassifierDataset:
    data: Data
    targets: torch.Tensor
    target_energy: torch.Tensor
