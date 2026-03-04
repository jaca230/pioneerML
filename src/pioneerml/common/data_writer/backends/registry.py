from __future__ import annotations

from collections.abc import Callable

from .base_output_backend import OutputBackend
from .parquet_output_backend import ParquetOutputBackend

_BackendFactory = Callable[[], OutputBackend]
_REGISTRY: dict[str, _BackendFactory] = {}


def register_output_backend(name: str, factory: _BackendFactory) -> None:
    key = str(name).strip().lower()
    if not key:
        raise ValueError("Backend name must be non-empty.")
    _REGISTRY[key] = factory


def create_output_backend(name: str) -> OutputBackend:
    key = str(name).strip().lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown output backend '{name}'. Available: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[key]()


def list_output_backends() -> list[str]:
    return sorted(_REGISTRY.keys())


register_output_backend("parquet", ParquetOutputBackend)
