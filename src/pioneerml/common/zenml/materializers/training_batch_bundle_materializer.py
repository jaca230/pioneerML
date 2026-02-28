from __future__ import annotations

from pathlib import Path

import torch
from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer

from pioneerml.common.loader import TrainingBatchBundle


class TrainingBatchBundleMaterializer(BaseMaterializer):
    """Materializer for generic training batch bundle artifacts."""

    SKIP_REGISTRATION = False
    ASSOCIATED_TYPES = (TrainingBatchBundle,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: type):
        path = Path(self.uri) / "batch.pt"
        payload = torch.load(path, weights_only=False, map_location="cpu")
        return TrainingBatchBundle(
            inputs=payload["inputs"],
            targets=payload.get("targets"),
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
        targets = getattr(dataset, "targets", None)
        try:
            if targets is not None:
                targets = targets.detach().cpu()
        except Exception:
            pass
        torch.save(
            {
                "inputs": inputs,
                "targets": targets,
                "loader": getattr(dataset, "loader", None),
                "loader_factory": getattr(dataset, "loader_factory", None),
                "metadata": dict(getattr(dataset, "metadata", {}) or {}),
            },
            path / "batch.pt",
        )

