from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import replace
from typing import TYPE_CHECKING

import pyarrow as pa

from pioneerml.common.data_loader.array_store.ndarray_store import NDArrayColumnSpec

if TYPE_CHECKING:
    from ..input_source_set import InputSourceSet


class InputBackend(ABC):
    """Backend contract for field planning and chunked reads."""

    @abstractmethod
    def schema_fields_intersection(self, sources: tuple[str, ...]) -> set[str]:
        raise NotImplementedError

    @abstractmethod
    def iter_tables(
        self,
        *,
        sources: tuple[str, ...],
        fields: list[str],
        row_groups_per_chunk: int,
    ) -> Iterator[pa.Table]:
        raise NotImplementedError

    def count_rows_per_source(self, *, sources: tuple[str, ...]) -> list[int]:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement row counting.")

    def resolve_dynamic_field_source_map(
        self,
        *,
        input_sources: InputSourceSet,
        required_fields: list[str],
        optional_fields: list[str] | None = None,
    ) -> dict[str, str]:
        resolved_optional_sources: dict[str, list[str]] = {
            str(name): list(paths) for name, paths in input_sources.optional_sources_by_name.items()
        }
        available_by_source: dict[str, set[str]] = {
            "main": self.schema_fields_intersection(input_sources.main_sources),
        }
        for source_name, source_paths in resolved_optional_sources.items():
            available_by_source[source_name] = self.schema_fields_intersection(tuple(source_paths))

        field_to_source: dict[str, str] = {}

        def assign_field(field_name: str, *, required: bool) -> None:
            if field_name in field_to_source:
                return
            if field_name in available_by_source["main"]:
                field_to_source[field_name] = "main"
                return
            for source_name in resolved_optional_sources:
                if field_name in available_by_source[source_name]:
                    field_to_source[field_name] = source_name
                    return
            if required:
                raise ValueError(f"Required field '{field_name}' not found in main or optional input sources.")

        for field in required_fields:
            assign_field(str(field), required=True)
        for field in optional_fields or []:
            assign_field(str(field), required=False)
        return field_to_source

    def resolve_declared_field_specs(
        self,
        *,
        input_sources: InputSourceSet,
        field_specs: tuple[NDArrayColumnSpec, ...],
        include_targets: bool,
    ) -> tuple[NDArrayColumnSpec, ...]:
        required_input_fields = [str(s.column) for s in field_specs if (not bool(s.target_only)) and bool(s.required)]
        required_target_fields = [str(s.column) for s in field_specs if bool(s.target_only) and bool(s.required)]
        optional_input_fields = [str(s.column) for s in field_specs if (not bool(s.target_only)) and (not bool(s.required))]
        required_fields = list(required_input_fields) + (list(required_target_fields) if include_targets else [])

        field_to_source = self.resolve_dynamic_field_source_map(
            input_sources=input_sources,
            required_fields=required_fields,
            optional_fields=optional_input_fields,
        )
        selected_fields = set(self.unique_fields(required_fields + optional_input_fields))
        resolved_specs: list[NDArrayColumnSpec] = []
        for spec in field_specs:
            field_name = str(spec.column)
            if field_name not in selected_fields or field_name not in field_to_source:
                continue
            resolved_specs.append(replace(spec, source=str(field_to_source[field_name])))
        return tuple(resolved_specs)

    @classmethod
    def fields_from_specs(
        cls,
        *,
        field_specs: tuple[NDArrayColumnSpec, ...],
        target_only: bool | None = None,
        required: bool | None = None,
        source: str | None = None,
    ) -> list[str]:
        fields: list[str] = []
        for spec in field_specs:
            if target_only is not None and bool(spec.target_only) != bool(target_only):
                continue
            if required is not None and bool(spec.required) != bool(required):
                continue
            if source is not None and str(spec.source) != str(source):
                continue
            fields.append(str(spec.column))
        return cls.unique_fields(fields)

    @staticmethod
    def unique_fields(fields: list[str]) -> list[str]:
        return list(dict.fromkeys(str(f) for f in fields))
