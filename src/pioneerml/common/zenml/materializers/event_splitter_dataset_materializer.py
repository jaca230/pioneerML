from __future__ import annotations

from pathlib import Path

import torch
from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer

try:
    from pioneerml.pipelines.training.event_splitting.dataset import (
        EventSplitterDataset,
    )
except Exception:  # pragma: no cover
    EventSplitterDataset = None


class EventSplitterDatasetMaterializer(BaseMaterializer):
    """Materializer for EventSplitterDataset artifacts."""

    SKIP_REGISTRATION = False
    ASSOCIATED_TYPES = (EventSplitterDataset,) if EventSplitterDataset is not None else ()
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: type):
        from pioneerml.pipelines.training.event_splitting.dataset import (
            EventSplitterDataset,
        )

        path = Path(self.uri) / "batch.pt"
        payload = torch.load(path, weights_only=False, map_location="cpu")
        return EventSplitterDataset(
            data=payload["data"],
            targets=payload["targets"],
            loader=payload.get("loader"),
            loader_factory=payload.get("loader_factory"),
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
                "loader_factory": getattr(dataset, "loader_factory", None),
            },
            path / "batch.pt",
        )
