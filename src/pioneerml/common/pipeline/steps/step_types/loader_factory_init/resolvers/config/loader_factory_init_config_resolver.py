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

    def resolve(self, *, cfg: dict[str, Any]) -> None:
        # Strict validation only; runtime objects are built in payload resolvers.
        input_sources_spec = self._require_dict(
            value=cfg.get("input_sources_spec"),
            context="loader_factory_init.input_sources_spec",
        )
        required_spec_keys = ("main_sources", "optional_sources_by_name", "source_type")
        missing = [k for k in required_spec_keys if k not in input_sources_spec]
        if missing:
            raise KeyError(f"loader_factory_init.input_sources_spec missing required keys: {missing}")

        main_sources = input_sources_spec.get("main_sources")
        optional_sources_by_name = input_sources_spec.get("optional_sources_by_name")
        source_type = SourceType.from_value(input_sources_spec.get("source_type"))
        if not isinstance(main_sources, list):
            raise TypeError("loader_factory_init.input_sources_spec.main_sources must be a list[str].")
        if optional_sources_by_name is not None and not isinstance(optional_sources_by_name, dict):
            raise TypeError(
                "loader_factory_init.input_sources_spec.optional_sources_by_name must be a dict[str, list[str]|None]."
            )
        for key, val in dict(optional_sources_by_name or {}).items():
            if not isinstance(key, str):
                raise TypeError("loader_factory_init.input_sources_spec.optional_sources_by_name keys must be strings.")
            if val is not None and not isinstance(val, list):
                raise TypeError(
                    "loader_factory_init.input_sources_spec.optional_sources_by_name values must be list[str] | None."
                )

        if "input_backend_name" not in cfg:
            raise KeyError("loader_factory_init missing required key: ['input_backend_name']")

        # Keep config normalized and JSON-safe.
        cfg["input_sources_spec"] = {
            "main_sources": [str(v) for v in main_sources],
            "optional_sources_by_name": (
                {str(k): v for k, v in (optional_sources_by_name or {}).items()} if optional_sources_by_name else {}
            ),
            "source_type": source_type.value,
        }
        cfg["input_backend_name"] = str(cfg.get("input_backend_name"))
