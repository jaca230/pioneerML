from __future__ import annotations

from pathlib import Path

import torch
from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer


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
        # Save on CPU to ensure device-agnostic loading.
        torch.save(tensor.cpu(), path / "tensor.pt")
