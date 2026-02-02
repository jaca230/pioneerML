from __future__ import annotations

from pathlib import Path

import torch
from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer


try:
    from pioneerml.zenml.pipelines.training.group_classification.batch import (
        GroupClassifierBatch,
    )
except Exception:  # pragma: no cover - optional import at runtime
    GroupClassifierBatch = None


class GroupClassifierBatchMaterializer(BaseMaterializer):
    """Materializer for GroupClassifierBatch artifacts."""

    SKIP_REGISTRATION = False
    ASSOCIATED_TYPES = (GroupClassifierBatch,) if GroupClassifierBatch is not None else ()
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: type):
        from pioneerml.zenml.pipelines.training.group_classification.batch import (
            GroupClassifierBatch,
        )

        path = Path(self.uri) / "batch.pt"
        payload = torch.load(path, weights_only=False, map_location="cpu")
        return GroupClassifierBatch(
            data=payload["data"],
            targets=payload["targets"],
            target_energy=payload["target_energy"],
        )

    def save(self, batch) -> None:
        path = Path(self.uri)
        path.mkdir(parents=True, exist_ok=True)
        data = batch.data
        try:
            data = data.to("cpu")
        except Exception:
            pass
        torch.save(
            {
                "data": data,
                "targets": batch.targets.detach().cpu(),
                "target_energy": batch.target_energy.detach().cpu(),
            },
            path / "batch.pt",
        )
