from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..ndarray_store import NDArrayColumnSpec


@dataclass(frozen=True)
class NDArrayField:
    """Typed field definition for extracted ndarray features/targets."""

    name: str
    column: str
    dtype: Any | None = None
    include_validity: bool = False
    required: bool = True
    target_only: bool = False

    def to_column_spec(self) -> NDArrayColumnSpec:
        return NDArrayColumnSpec(
            column=str(self.column),
            field=str(self.name),
            dtype=self.dtype,
            target_only=bool(self.target_only),
            include_validity=bool(self.include_validity),
            required=bool(self.required),
        )
