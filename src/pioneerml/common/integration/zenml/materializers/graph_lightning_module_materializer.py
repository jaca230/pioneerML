from __future__ import annotations

from pathlib import Path

import torch
from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer

from pioneerml.common.pipeline.steps.training.utils import GraphLightningModule


class GraphLightningModuleMaterializer(BaseMaterializer):
    """Materializer for GraphLightningModule artifacts."""

    SKIP_REGISTRATION = False
    ASSOCIATED_TYPES = (GraphLightningModule,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.MODEL

    def load(self, data_type: type):
        _ = data_type
        path = Path(self.uri) / "module.pt"
        return torch.load(path, weights_only=False, map_location="cpu")

    def save(self, module: GraphLightningModule) -> None:
        path = Path(self.uri)
        path.mkdir(parents=True, exist_ok=True)
        try:
            module = module.to("cpu")
        except Exception:
            pass
        torch.save(module, path / "module.pt")
