from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .backends.base_backend import InputBackend
from .backends.parquet_backend import ParquetInputBackend
from .factory import InputBackendFactory, REGISTRY


def register_input_backend(name: str, backend_cls: type[InputBackend]) -> None:
    REGISTRY.register(str(name))(backend_cls)


def create_input_backend(name: str, *, config: Mapping[str, Any] | None = None) -> InputBackend:
    return InputBackendFactory(
        backend_name=str(name),
        config=dict(config or {}),
    ).build()


def list_input_backends() -> list[str]:
    return list(REGISTRY.list())


# Keep import side-effect explicit for built-ins in this namespace.
_ = ParquetInputBackend

