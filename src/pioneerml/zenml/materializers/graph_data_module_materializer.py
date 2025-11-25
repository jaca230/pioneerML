from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer

from pioneerml.training.datamodules.base import GraphDataModule


class GraphDataModuleMaterializer(BaseMaterializer):
    """Materializer for GraphDataModule instances used in tutorials."""

    SKIP_REGISTRATION = False
    ASSOCIATED_TYPES = (GraphDataModule,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: type) -> Any:
        path = Path(self.uri)
        return torch.load(path / "datamodule.pt", weights_only=False)

    def save(self, datamodule: Any) -> None:
        path = Path(self.uri)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(datamodule, path / "datamodule.pt")
