from __future__ import annotations

from dataclasses import dataclass

from ..ndarray_store import NDArrayColumnSpec


@dataclass(frozen=True)
class TargetSchema:
    fields: tuple[NDArrayColumnSpec, ...]

    def required_columns(self) -> list[str]:
        return list(dict.fromkeys(str(f.column) for f in self.fields if bool(f.target_only) and bool(f.required)))

    def field_names(self) -> tuple[str, ...]:
        return tuple(str(f.field) for f in self.fields if bool(f.target_only))

    def to_column_specs(self) -> tuple[NDArrayColumnSpec, ...]:
        return tuple(f for f in self.fields if bool(f.target_only))
