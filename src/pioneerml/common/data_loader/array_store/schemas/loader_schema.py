from __future__ import annotations

from dataclasses import dataclass

from ..ndarray_store import NDArrayColumnSpec
from .feature_schema import FeatureSchema
from .target_schema import TargetSchema


@dataclass(frozen=True)
class LoaderSchema:
    features: FeatureSchema
    targets: TargetSchema | None = None

    def required_columns(self, *, include_targets: bool) -> list[str]:
        cols = list(self.features.required_columns())
        if include_targets and self.targets is not None:
            cols.extend(self.targets.required_columns())
        return list(dict.fromkeys(cols))

    def to_column_specs(self, *, include_targets: bool) -> tuple[NDArrayColumnSpec, ...]:
        specs = list(self.features.to_column_specs())
        if include_targets and self.targets is not None:
            specs.extend(self.targets.to_column_specs())
        return tuple(specs)

    def has_field(self, name: str, *, include_targets: bool = True) -> bool:
        key = str(name)
        if key in set(self.features.field_names()):
            return True
        if include_targets and self.targets is not None and key in set(self.targets.field_names()):
            return True
        return False

    def validate_required_fields(self, required_fields: list[str], *, include_targets: bool) -> None:
        missing = [f for f in required_fields if not self.has_field(f, include_targets=include_targets)]
        if missing:
            raise ValueError(f"Schema missing required fields: {missing}")
