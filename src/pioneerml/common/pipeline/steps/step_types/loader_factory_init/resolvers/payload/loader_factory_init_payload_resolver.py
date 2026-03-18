from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pioneerml.common.data_loader.factory import LoaderFactory
from pioneerml.common.data_loader.input_source import InputSourceSet, SourceType

from .....resolver import BasePayloadResolver
from ..config.loader_factory_init_config_resolver import LoaderFactoryInitConfigResolver


class LoaderFactoryInitPayloadResolver(BasePayloadResolver):
    def resolve(
        self,
        *,
        payloads: Mapping[str, Any] | None,
        runtime_state: dict[str, Any],
    ) -> None:
        input_sources_spec = self._resolve_input_sources_spec(payloads=payloads)
        main_sources = list(input_sources_spec["main_sources"])
        optional_sources_by_name = dict(input_sources_spec.get("optional_sources_by_name") or {})
        source_type = SourceType.from_value(input_sources_spec["source_type"])
        input_backend_name = str(self.step.config_json["input_backend_name"])
        loader_name = str(self.step.config_json["loader_name"])

        input_sources = InputSourceSet(
            main_sources=[str(v) for v in main_sources],
            optional_sources_by_name={str(k): v for k, v in optional_sources_by_name.items()},
            source_type=source_type,
        )
        runtime_state["input_sources"] = input_sources
        runtime_state["loader_factory"] = LoaderFactory(
            loader_name=loader_name,
            input_sources=input_sources,
            input_backend_name=input_backend_name,
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
        return dict(self.step.config_json["input_sources_spec"])
