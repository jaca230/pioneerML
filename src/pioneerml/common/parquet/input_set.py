from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq


@dataclass(frozen=True)
class ParquetInputSet:
    """Aligned parquet source set with one required main source and optional aligned sources."""

    main_paths: tuple[str, ...]
    optional_paths_by_name: dict[str, tuple[str, ...]]

    @staticmethod
    def resolve_required_paths(paths: list[str], *, label: str) -> tuple[str, ...]:
        resolved = tuple(str(Path(p).expanduser().resolve()) for p in paths)
        if not resolved:
            raise RuntimeError(f"No {label} provided.")
        missing = [p for p in resolved if not Path(p).exists()]
        if missing:
            raise RuntimeError(f"Missing {label} path(s): {missing}")
        return resolved

    @classmethod
    def resolve_optional_aligned_paths(
        cls,
        optional_paths: list[str] | None,
        *,
        label: str,
        aligned_to: tuple[str, ...],
    ) -> tuple[str, ...] | None:
        if optional_paths is None:
            return None
        resolved = cls.resolve_required_paths(optional_paths, label=label)
        if len(resolved) != len(aligned_to):
            raise ValueError(
                f"{label} must match parquet_paths length. "
                f"Got {len(resolved)} vs {len(aligned_to)}."
            )
        return resolved

    @staticmethod
    def schema_columns_intersection(parquet_paths: tuple[str, ...]) -> set[str]:
        if not parquet_paths:
            return set()
        intersection: set[str] | None = None
        for path in parquet_paths:
            names = set(pq.read_schema(path).names)
            intersection = names if intersection is None else (intersection & names)
            if not intersection:
                break
        return set(intersection or set())

    @classmethod
    def from_paths(
        cls,
        *,
        main_paths: list[str],
        optional_paths_by_name: dict[str, list[str] | None] | None = None,
    ) -> ParquetInputSet:
        return cls(main_paths=main_paths, optional_paths_by_name=optional_paths_by_name)

    def __init__(
        self,
        *,
        main_paths: list[str],
        optional_paths_by_name: dict[str, list[str] | None] | None = None,
    ) -> None:
        resolved_main = self.resolve_required_paths(main_paths, label="parquet_paths")
        resolved_optional: dict[str, tuple[str, ...]] = {}
        for name, paths in dict(optional_paths_by_name or {}).items():
            if paths is None:
                continue
            key = str(name)
            resolved = self.resolve_optional_aligned_paths(
                paths,
                label=f"{key}_parquet_paths",
                aligned_to=resolved_main,
            )
            if resolved is not None:
                resolved_optional[key] = resolved

        object.__setattr__(self, "main_paths", resolved_main)
        object.__setattr__(self, "optional_paths_by_name", resolved_optional)

    def source_paths(self, source_name: str) -> tuple[str, ...] | None:
        return self.optional_paths_by_name.get(str(source_name))

    def with_optional_paths(self, source_name: str, optional_paths: list[str] | None) -> ParquetInputSet:
        key = str(source_name)
        updated: dict[str, list[str] | None] = self.as_dynamic_source_map()
        updated[key] = optional_paths
        return ParquetInputSet(
            main_paths=list(self.main_paths),
            optional_paths_by_name=updated,
        )

    def with_optional_paths_map(self, optional_paths_by_name: dict[str, list[str] | None]) -> ParquetInputSet:
        updated: dict[str, list[str] | None] = self.as_dynamic_source_map()
        updated.update({str(k): v for k, v in dict(optional_paths_by_name).items()})
        return ParquetInputSet(
            main_paths=list(self.main_paths),
            optional_paths_by_name=updated,
        )

    def as_dynamic_source_map(self) -> dict[str, list[str] | None]:
        return {
            str(name): (list(paths) if paths else None)
            for name, paths in self.optional_paths_by_name.items()
        }

