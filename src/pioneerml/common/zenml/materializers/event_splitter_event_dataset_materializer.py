from __future__ import annotations

from pathlib import Path

import torch
from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer


class EventSplitterEventDatasetMaterializer(BaseMaterializer):
    """Materializer for EventSplitterEventDataset artifacts."""

    SKIP_REGISTRATION = True
    ASSOCIATED_TYPES = (object,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: type):
        from pioneerml.pipelines.training.event_splitter_event.dataset import (
            EventSplitterEventDataset,
        )

        path = Path(self.uri) / "batch.pt"
        payload = torch.load(path, weights_only=False, map_location="cpu")
        return EventSplitterEventDataset(
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
