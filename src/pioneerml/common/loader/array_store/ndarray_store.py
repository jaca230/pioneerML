from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NDArrayColumnSpec:
    """Declarative spec for Arrow column -> ndarray extraction."""

    column: str
    field: str
    dtype: Any | None = None
    target_only: bool = False
    include_validity: bool = False
    required: bool = True


class NDArrayStore:
    """Thin wrapper around canonical extracted ndarray mappings."""

    def __init__(self, initial: Mapping[str, np.ndarray] | None = None) -> None:
        self._data: dict[str, np.ndarray] = dict(initial or {})

    @staticmethod
    def values_key(name: str) -> str:
        return f"{name}/values"

    @staticmethod
    def offsets_key(name: str, level: int = 0) -> str:
        return f"{name}/offsets/{int(level)}"

    @staticmethod
    def shape_key(name: str) -> str:
        return f"{name}/shape"

    @staticmethod
    def valid_key(name: str) -> str:
        return f"{name}/valid"

    def raw(self) -> dict[str, np.ndarray]:
        return self._data

    def set_raw(self, key: str, value: np.ndarray) -> None:
        self._data[str(key)] = value

    def get_raw(self, key: str, default: Any = None) -> Any:
        return self._data.get(str(key), default)

    def has_raw(self, key: str) -> bool:
        return str(key) in self._data

    def update(self, mapping: Mapping[str, np.ndarray]) -> None:
        self._data.update(mapping)

    def set_values(self, name: str, value: np.ndarray) -> None:
        self._data[self.values_key(name)] = value

    def set_offsets(self, name: str, level: int, value: np.ndarray) -> None:
        self._data[self.offsets_key(name, level)] = value

    def set_shape(self, name: str, value: np.ndarray) -> None:
        self._data[self.shape_key(name)] = value

    def set_valid(self, name: str, value: np.ndarray) -> None:
        self._data[self.valid_key(name)] = value

    def values(self, name: str) -> np.ndarray:
        return self._data[self.values_key(name)]

    def offsets(self, name: str, level: int = 0) -> np.ndarray:
        return self._data[self.offsets_key(name, level)]

    def shape(self, name: str) -> np.ndarray:
        return self._data[self.shape_key(name)]

    def valid(self, name: str) -> np.ndarray:
        return self._data[self.valid_key(name)]
