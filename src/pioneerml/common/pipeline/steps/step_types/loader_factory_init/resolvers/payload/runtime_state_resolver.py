from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pioneerml.common.data_loader.loaders.factory import LoaderFactory
from pioneerml.common.data_loader.loaders.input_source import InputSourceSet, SourceType

from .....resolver import BasePayloadResolver
from ..config.config_resolver import LoaderFactoryInitConfigResolver


class LoaderFactoryInitStateResolver(BasePayloadResolver):
    def resolve(
        self,
        *,
        payloads: Mapping[str, Any] | None,
        runtime_state: dict[str, Any],
    ) -> None:
        loader_block = dict(self.step.config_json.get("loader") or {})
        loader_cfg = dict(loader_block.get("config") or {})
        input_sources_spec = self._resolve_input_sources_spec(payloads=payloads)
        main_sources = list(input_sources_spec["main_sources"])
        optional_sources_by_name = dict(input_sources_spec.get("optional_sources_by_name") or {})
        source_type = SourceType.from_value(input_sources_spec["source_type"])
        input_backend_name = str(loader_cfg["input_backend_name"])
        loader_name = str(loader_block["type"])
        mode = str(loader_cfg.get("mode", "train"))

        input_sources = InputSourceSet(
            main_sources=[str(v) for v in main_sources],
            optional_sources_by_name={str(k): v for k, v in optional_sources_by_name.items()},
            source_type=source_type,
        )
        runtime_state["input_sources"] = input_sources
        runtime_state["loader_factory"] = LoaderFactory(
            loader_name=loader_name,
            config={
                "input_sources": input_sources,
                "input_backend_name": input_backend_name,
                "mode": mode,
            },
        )

    def _resolve_input_sources_spec(self, *, payloads: Mapping[str, Any] | None) -> dict[str, Any]:
        if isinstance(payloads, Mapping):
            raw = payloads.get("input_source_set")
            if raw is None:
                raw = payloads.get("input_sources_spec")
            if raw is not None:
                return LoaderFactoryInitConfigResolver.normalize_input_sources_spec(
                    value=raw,
                    context=f"{self.step.__class__.__name__}.payloads.input_source_set",
                )
        loader_block = dict(self.step.config_json.get("loader") or {})
        loader_cfg = dict(loader_block.get("config") or {})
        return dict(loader_cfg["input_sources_spec"])
