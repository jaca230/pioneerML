from __future__ import annotations

from pathlib import Path

import torch
from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer

from pioneerml.data_loader import BatchBundle


class BatchBundleMaterializer(BaseMaterializer):
    """Materializer for generic batch bundle artifacts."""

    SKIP_REGISTRATION = False
    ASSOCIATED_TYPES = (BatchBundle,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: type):
        path = Path(self.uri) / "batch.pt"
        payload = torch.load(path, weights_only=False, map_location="cpu")
        return BatchBundle(
            data=payload["data"],
            loader=payload.get("loader"),
            loader_factory=payload.get("loader_factory"),
            metadata=dict(payload.get("metadata") or {}),
        )

    def save(self, dataset) -> None:
        path = Path(self.uri)
        path.mkdir(parents=True, exist_ok=True)
        data = dataset.data
        try:
            data = data.to("cpu")
        except Exception:
            pass
        torch.save(
            {
                "data": data,
                "loader": getattr(dataset, "loader", None),
                "loader_factory": getattr(dataset, "loader_factory", None),
                "metadata": dict(getattr(dataset, "metadata", {}) or {}),
            },
            path / "batch.pt",
        )

