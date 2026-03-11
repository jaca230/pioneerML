from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pioneerml.common.data_loader.factory import LoaderFactory
from pioneerml.common.data_loader.input_source import InputSourceSet, SourceType

from .....resolver import BasePayloadResolver


class LoaderFactoryInitPayloadResolver(BasePayloadResolver):
    def resolve(
        self,
        *,
        payloads: Mapping[str, Any] | None,
        runtime_state: dict[str, Any],
    ) -> None:
        _ = payloads
        input_sources_spec = dict(self.step.config_json["input_sources_spec"])
        main_sources = list(input_sources_spec["main_sources"])
        optional_sources_by_name = dict(input_sources_spec.get("optional_sources_by_name") or {})
        source_type = SourceType.from_value(input_sources_spec["source_type"])
        input_backend_name = str(self.step.config_json["input_backend_name"])

        loader_name_fn = getattr(self.step, "loader_name", None)
        if not callable(loader_name_fn):
            raise RuntimeError(f"{self.step.__class__.__name__} must implement loader_name().")
        loader_name = str(loader_name_fn())

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
