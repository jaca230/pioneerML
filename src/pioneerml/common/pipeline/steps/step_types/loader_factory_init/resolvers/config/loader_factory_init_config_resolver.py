from __future__ import annotations

from typing import Any

from pioneerml.common.data_loader.input_source import SourceType

from .....resolver import BaseConfigResolver


class LoaderFactoryInitConfigResolver(BaseConfigResolver):
    @staticmethod
    def _require_dict(*, value: Any, context: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise TypeError(f"{context} must be a dict.")
        return dict(value)

    @classmethod
    def normalize_input_sources_spec(cls, *, value: Any, context: str) -> dict[str, Any]:
        spec = cls._require_dict(value=value, context=context)
        main_sources = spec.get("main_sources")
        if main_sources is None:
            main_sources = spec.get("main_paths")
        optional_sources_by_name = spec.get("optional_sources_by_name")
        if optional_sources_by_name is None:
            optional_sources_by_name = spec.get("optional_paths_by_name")
        source_type = SourceType.from_value(spec.get("source_type", "file"))

        if not isinstance(main_sources, list):
            raise TypeError(f"{context}.main_sources (or main_paths) must be a list[str].")
        if optional_sources_by_name is None:
            optional_sources_by_name = {}
        if not isinstance(optional_sources_by_name, dict):
            raise TypeError(
                f"{context}.optional_sources_by_name (or optional_paths_by_name) must be a dict[str, list[str]|None]."
            )
        for key, val in dict(optional_sources_by_name).items():
            if not isinstance(key, str):
                raise TypeError(f"{context}.optional_sources_by_name keys must be strings.")
            if val is not None and not isinstance(val, list):
                raise TypeError(f"{context}.optional_sources_by_name values must be list[str] | None.")

        return {
            "main_sources": [str(v) for v in main_sources],
            "optional_sources_by_name": {str(k): v for k, v in dict(optional_sources_by_name).items()},
            "source_type": source_type.value,
        }

    def resolve(self, *, cfg: dict[str, Any]) -> None:
        # Strict validation only; runtime objects are built in payload resolvers.
        loader_name = cfg.get("loader_name")
        if not isinstance(loader_name, str):
            raise KeyError("loader_factory_init missing required string key: ['loader_name']")
        loader_name = loader_name.strip()
        if loader_name == "":
            raise ValueError("loader_factory_init.loader_name cannot be empty.")

        input_sources_spec = self.normalize_input_sources_spec(
            value=cfg.get("input_sources_spec"),
            context="loader_factory_init.input_sources_spec",
        )

        if "input_backend_name" not in cfg:
            raise KeyError("loader_factory_init missing required key: ['input_backend_name']")

        # Keep config normalized and JSON-safe.
        cfg["loader_name"] = loader_name
        cfg["input_sources_spec"] = input_sources_spec
        cfg["input_backend_name"] = str(cfg.get("input_backend_name"))
