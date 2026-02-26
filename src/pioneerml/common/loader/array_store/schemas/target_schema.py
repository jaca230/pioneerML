from __future__ import annotations

from dataclasses import dataclass

from ..ndarray_store import NDArrayColumnSpec
from .ndarray_field import NDArrayField


@dataclass(frozen=True)
class TargetSchema:
    fields: tuple[NDArrayField, ...]

    def required_columns(self) -> list[str]:
        return list(dict.fromkeys(str(f.column) for f in self.fields if bool(f.required)))

    def field_names(self) -> tuple[str, ...]:
        return tuple(str(f.name) for f in self.fields)

    def to_column_specs(self) -> tuple[NDArrayColumnSpec, ...]:
        return tuple(f.to_column_spec() for f in self.fields)
