"""
Lightweight ZenML materializers to silence pickle warnings in tutorials.

These materializers keep artifacts tiny and easy to reload for the
synthetic tutorial pipelines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data
from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.utils import source_utils
from pioneerml.training.datamodules.base import GraphDataModule

# Ensure ZenML resolves sources relative to the src/ directory (avoid `src.` prefixes).
source_utils.set_custom_source_root(Path(__file__).resolve().parents[2])


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


class TorchTensorMaterializer(BaseMaterializer):
    """Materializer for torch.Tensor objects to avoid pickle warnings."""

    SKIP_REGISTRATION = False
    ASSOCIATED_TYPES = (torch.Tensor,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: type) -> torch.Tensor:
        path = Path(self.uri)
        return torch.load(path / "tensor.pt", weights_only=False, map_location="cpu")

    def save(self, tensor: torch.Tensor) -> None:
        path = Path(self.uri)
        path.mkdir(parents=True, exist_ok=True)
        # Save on CPU to ensure device-agnostic loading
        torch.save(tensor.cpu(), path / "tensor.pt")


# Ensure the module path resolves cleanly when ZenML constructs sources.
PyGDataListMaterializer.__module__ = "pioneerml.zenml.materializers"
GraphDataModuleMaterializer.__module__ = "pioneerml.zenml.materializers"
TorchTensorMaterializer.__module__ = "pioneerml.zenml.materializers"
