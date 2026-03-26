from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .factory import OutputBackendFactory, REGISTRY
from .base_output_backend import OutputBackend
from .parquet_output_backend import ParquetOutputBackend


def register_output_backend(name: str, backend_cls: type[OutputBackend]) -> None:
    REGISTRY.register(str(name))(backend_cls)


def create_output_backend(name: str, *, config: Mapping[str, Any] | None = None) -> OutputBackend:
    return OutputBackendFactory(
        backend_name=str(name),
        config=dict(config or {}),
    ).build()


def list_output_backends() -> list[str]:
    return list(REGISTRY.list())


# Keep import side-effect explicit for built-ins in this namespace.
_ = ParquetOutputBackend

