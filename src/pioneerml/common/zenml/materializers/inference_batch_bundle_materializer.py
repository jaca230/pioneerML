from __future__ import annotations

from pathlib import Path

import torch
from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer

from pioneerml.common.loader import InferenceBatchBundle


class InferenceBatchBundleMaterializer(BaseMaterializer):
    """Materializer for generic inference batch bundle artifacts."""

    SKIP_REGISTRATION = False
    ASSOCIATED_TYPES = (InferenceBatchBundle,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: type):
        path = Path(self.uri) / "batch.pt"
        payload = torch.load(path, weights_only=False, map_location="cpu")
        return InferenceBatchBundle(
            inputs=payload["inputs"],
            ids=payload.get("ids"),
            loader=payload.get("loader"),
            loader_factory=payload.get("loader_factory"),
            metadata=dict(payload.get("metadata") or {}),
        )

    def save(self, dataset) -> None:
        path = Path(self.uri)
        path.mkdir(parents=True, exist_ok=True)
        inputs = dataset.inputs
        try:
            inputs = inputs.to("cpu")
        except Exception:
            pass
        ids = getattr(dataset, "ids", None)
        try:
            if ids is not None:
                ids = ids.detach().cpu()
        except Exception:
            pass
        torch.save(
            {
                "inputs": inputs,
                "ids": ids,
                "loader": getattr(dataset, "loader", None),
                "loader_factory": getattr(dataset, "loader_factory", None),
                "metadata": dict(getattr(dataset, "metadata", {}) or {}),
            },
            path / "batch.pt",
        )

