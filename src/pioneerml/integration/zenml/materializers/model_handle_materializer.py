from __future__ import annotations

import json
from pathlib import Path

from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer

from pioneerml.integration.pytorch.model_handles import BaseModelHandle
from pioneerml.integration.pytorch.model_handles import ModelHandleFactory


class ModelHandleMaterializer(BaseMaterializer):
    """Materializer for model handles to avoid Pickle fallback warnings."""

    SKIP_REGISTRATION = False
    ASSOCIATED_TYPES = (BaseModelHandle,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.MODEL

    def load(self, data_type: type) -> BaseModelHandle:
        _ = data_type
        payload_path = Path(self.uri) / "model_handle.json"
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("Invalid model handle materializer payload.")

        handle_type = str(payload.get("type") or "").strip()
        model_path = payload.get("model_path")
        if handle_type == "":
            raise ValueError("Model handle payload missing non-empty 'type'.")
        if not isinstance(model_path, str) or model_path.strip() == "":
            raise ValueError("Model handle payload missing non-empty 'model_path'.")

        return ModelHandleFactory(model_type=handle_type).build(
            config={"model_path": str(model_path)}
        )

    def save(self, model_handle: BaseModelHandle) -> None:
        path = Path(self.uri)
        path.mkdir(parents=True, exist_ok=True)
        payload = dict(model_handle.to_payload())
        (path / "model_handle.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
