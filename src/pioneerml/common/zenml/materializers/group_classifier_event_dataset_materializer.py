from __future__ import annotations

from pathlib import Path

import torch
from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer

try:
    from pioneerml.pipelines.training.group_classification_event.dataset import (
        GroupClassifierEventDataset,
    )
except Exception:  # pragma: no cover
    GroupClassifierEventDataset = None


class GroupClassifierEventDatasetMaterializer(BaseMaterializer):
    """Materializer for GroupClassifierEventDataset artifacts."""

    SKIP_REGISTRATION = False
    ASSOCIATED_TYPES = (GroupClassifierEventDataset,) if GroupClassifierEventDataset is not None else ()
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: type):
        from pioneerml.pipelines.training.group_classification_event.dataset import (
            GroupClassifierEventDataset,
        )

        path = Path(self.uri) / "batch.pt"
        payload = torch.load(path, weights_only=False, map_location="cpu")
        return GroupClassifierEventDataset(
            data=payload["data"],
            targets=payload["targets"],
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
                "targets": dataset.targets.detach().cpu(),
            },
            path / "batch.pt",
        )
