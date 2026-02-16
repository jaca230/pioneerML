from __future__ import annotations

from pathlib import Path

import torch
from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer

try:
    from pioneerml.pipelines.training.group_splitting.dataset import (
        GroupSplitterDataset,
    )
except Exception:  # pragma: no cover
    GroupSplitterDataset = None


class GroupSplitterDatasetMaterializer(BaseMaterializer):
    """Materializer for GroupSplitterDataset artifacts."""

    SKIP_REGISTRATION = False
    ASSOCIATED_TYPES = (GroupSplitterDataset,) if GroupSplitterDataset is not None else ()
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: type):
        from pioneerml.pipelines.training.group_splitting.dataset import (
            GroupSplitterDataset,
        )

        path = Path(self.uri) / "batch.pt"
        payload = torch.load(path, weights_only=False, map_location="cpu")
        return GroupSplitterDataset(
            data=payload["data"],
            targets=payload["targets"],
            loader=payload.get("loader"),
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
                "loader": getattr(dataset, "loader", None),
            },
            path / "batch.pt",
        )
