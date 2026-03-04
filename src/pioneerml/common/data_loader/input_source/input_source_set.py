from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InputSourceSet:
    """Aligned source set with one required main source list and optional aligned source lists."""

    main_sources: tuple[str, ...]
    optional_sources_by_name: dict[str, tuple[str, ...]]

    @staticmethod
    def resolve_required_sources(sources: list[str], *, label: str) -> tuple[str, ...]:
        resolved = tuple(str(Path(s).expanduser().resolve()) for s in sources)
        if not resolved:
            raise RuntimeError(f"No {label} provided.")
        missing = [s for s in resolved if not Path(s).exists()]
        if missing:
            raise RuntimeError(f"Missing {label} source(s): {missing}")
        return resolved

    @classmethod
    def resolve_optional_aligned_sources(
        cls,
        optional_sources: list[str] | None,
        *,
        label: str,
        aligned_to: tuple[str, ...],
    ) -> tuple[str, ...] | None:
        if optional_sources is None:
            return None
        resolved = cls.resolve_required_sources(optional_sources, label=label)
        if len(resolved) != len(aligned_to):
            raise ValueError(
                f"{label} must match main_sources length. "
                f"Got {len(resolved)} vs {len(aligned_to)}."
            )
        return resolved

    @classmethod
    def from_sources(
        cls,
        *,
        main_sources: list[str],
        optional_sources_by_name: dict[str, list[str] | None] | None = None,
    ) -> InputSourceSet:
        return cls(main_sources=main_sources, optional_sources_by_name=optional_sources_by_name)

    def __init__(
        self,
        *,
        main_sources: list[str],
        optional_sources_by_name: dict[str, list[str] | None] | None = None,
    ) -> None:
        resolved_main = self.resolve_required_sources(main_sources, label="main_sources")
        resolved_optional: dict[str, tuple[str, ...]] = {}
        for name, sources in dict(optional_sources_by_name or {}).items():
            if sources is None:
                continue
            key = str(name)
            resolved = self.resolve_optional_aligned_sources(
                sources,
                label=f"{key}_sources",
                aligned_to=resolved_main,
            )
            if resolved is not None:
                resolved_optional[key] = resolved

        object.__setattr__(self, "main_sources", resolved_main)
        object.__setattr__(self, "optional_sources_by_name", resolved_optional)

    def source_entries(self, source_name: str) -> tuple[str, ...] | None:
        return self.optional_sources_by_name.get(str(source_name))

    def with_optional_sources(self, source_name: str, optional_sources: list[str] | None) -> InputSourceSet:
        key = str(source_name)
        updated: dict[str, list[str] | None] = self.as_dynamic_source_map()
        updated[key] = optional_sources
        return InputSourceSet(
            main_sources=list(self.main_sources),
            optional_sources_by_name=updated,
        )

    def with_optional_sources_map(self, optional_sources_by_name: dict[str, list[str] | None]) -> InputSourceSet:
        updated: dict[str, list[str] | None] = self.as_dynamic_source_map()
        updated.update({str(k): v for k, v in dict(optional_sources_by_name).items()})
        return InputSourceSet(
            main_sources=list(self.main_sources),
            optional_sources_by_name=updated,
        )

    def as_dynamic_source_map(self) -> dict[str, list[str] | None]:
        return {
            str(name): (list(sources) if sources else None)
            for name, sources in self.optional_sources_by_name.items()
        }

