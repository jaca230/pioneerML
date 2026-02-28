from __future__ import annotations

from typing import Any


class BaseBatchBundle:
    def __init__(
        self,
        *,
        inputs: Any,
        loader: Any | None = None,
        loader_factory: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.inputs = inputs
        self.loader = loader
        self.loader_factory = loader_factory
        self.metadata = dict(metadata or {})
