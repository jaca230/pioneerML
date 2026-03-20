from __future__ import annotations

from collections.abc import Callable

from .backends import InputBackend, ParquetInputBackend

_BackendFactory = Callable[[], InputBackend]
_BACKEND_REGISTRY: dict[str, _BackendFactory] = {}


def register_input_backend(name: str, factory: _BackendFactory) -> None:
    key = str(name).strip().lower()
    if not key:
        raise ValueError("Backend name must be non-empty.")
    _BACKEND_REGISTRY[key] = factory


def create_input_backend(name: str) -> InputBackend:
    key = str(name).strip().lower()
    if key not in _BACKEND_REGISTRY:
        available = sorted(_BACKEND_REGISTRY.keys())
        raise ValueError(f"Unknown input backend '{name}'. Available backends: {available}")
    return _BACKEND_REGISTRY[key]()


def list_input_backends() -> list[str]:
    return sorted(_BACKEND_REGISTRY.keys())


register_input_backend("parquet", ParquetInputBackend)
