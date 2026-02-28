from __future__ import annotations

from typing import Any

from .base_batch_bundle import BaseBatchBundle

class InferenceBatchBundle(BaseBatchBundle):
    def __init__(
        self,
        *,
        inputs: Any,
        ids: Any | None = None,
        loader: Any | None = None,
        loader_factory: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            inputs=inputs,
            loader=loader,
            loader_factory=loader_factory,
            metadata=metadata,
        )
        self.ids = ids
