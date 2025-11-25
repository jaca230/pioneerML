from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Data
from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer


class PyGDataListMaterializer(BaseMaterializer):
    """Materializer for lists of torch_geometric Data objects."""

    SKIP_REGISTRATION = False
    ASSOCIATED_TYPES = (list,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: type) -> list[Data]:
        path = Path(self.uri)
        return torch.load(path / "data.pt", weights_only=False)

    def save(self, data: list[Data]) -> None:
        path = Path(self.uri)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(data, path / "data.pt")
