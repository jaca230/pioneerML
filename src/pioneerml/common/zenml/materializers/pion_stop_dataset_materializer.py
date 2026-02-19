from __future__ import annotations

from pathlib import Path

import torch
from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer

try:
    from pioneerml.pipelines.training.pion_stop.dataset import PionStopDataset
except Exception:  # pragma: no cover
    PionStopDataset = None


class PionStopDatasetMaterializer(BaseMaterializer):
    """Materializer for PionStopDataset artifacts."""

    SKIP_REGISTRATION = False
    ASSOCIATED_TYPES = (PionStopDataset,) if PionStopDataset is not None else ()
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: type):
        from pioneerml.pipelines.training.pion_stop.dataset import PionStopDataset

        path = Path(self.uri) / "batch.pt"
        payload = torch.load(path, weights_only=False, map_location="cpu")
        return PionStopDataset(
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
