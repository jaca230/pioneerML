from __future__ import annotations

from pathlib import Path

import torch
from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer


try:
    from pioneerml.pipelines.training.group_classification.dataset import (
        GroupClassifierDataset,
    )
except Exception:  # pragma: no cover - optional import at runtime
    GroupClassifierDataset = None


class GroupClassifierDatasetMaterializer(BaseMaterializer):
    """Materializer for GroupClassifierDataset artifacts."""

    SKIP_REGISTRATION = False
    ASSOCIATED_TYPES = (GroupClassifierDataset,) if GroupClassifierDataset is not None else ()
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: type):
        from pioneerml.pipelines.training.group_classification.dataset import (
            GroupClassifierDataset,
        )

        path = Path(self.uri) / "batch.pt"
        payload = torch.load(path, weights_only=False, map_location="cpu")
        return GroupClassifierDataset(
            data=payload["data"],
            targets=payload["targets"],
            target_energy=payload["target_energy"],
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
                "target_energy": dataset.target_energy.detach().cpu(),
            },
            path / "batch.pt",
        )
