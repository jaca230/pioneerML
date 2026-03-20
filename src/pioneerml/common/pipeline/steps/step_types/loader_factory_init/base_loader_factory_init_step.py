from __future__ import annotations

from pioneerml.common.data_loader.loaders.factory import LoaderFactory
from pioneerml.common.data_loader.loaders.input_source import InputSourceSet
from .payloads import LoaderFactoryInitStepPayload
from .resolvers import (
    LoaderFactoryInitConfigResolver,
    LoaderFactoryInitStateResolver,
)

from ..base_pipeline_step import BasePipelineStep


class BaseLoaderFactoryInitStep(BasePipelineStep):
    DEFAULT_CONFIG = {
        "loader": {
            "type": "required",
            "config": {
                "input_sources_spec": {
                    "main_sources": [],
                    "optional_sources_by_name": {},
                    "source_type": "file",
                },
                "input_backend_name": "parquet",
                "mode": "train",
            },
        },
    }
    config_resolver_classes = (LoaderFactoryInitConfigResolver,)
    payload_resolver_classes = (LoaderFactoryInitStateResolver,)

    def _build_payload(
        self,
        *,
        loader_factory: LoaderFactory,
        input_sources: InputSourceSet,
        input_backend_name: str,
    ) -> LoaderFactoryInitStepPayload:
        return LoaderFactoryInitStepPayload(
            loader_factory=loader_factory,
            input_sources=input_sources,
            input_backend_name=input_backend_name,
        )

    def _execute(self) -> LoaderFactoryInitStepPayload:
        loader_factory = self.runtime_state.get("loader_factory")
        input_sources = self.runtime_state.get("input_sources")
        loader_cfg = dict(dict(self.config_json.get("loader") or {}).get("config") or {})
        input_backend_name = str(loader_cfg.get("input_backend_name", "parquet"))
        if not isinstance(loader_factory, LoaderFactory):
            raise RuntimeError("Loader factory-init resolver did not provide a valid loader_factory.")
        if not isinstance(input_sources, InputSourceSet):
            raise RuntimeError("Loader factory-init resolver did not provide valid input_sources.")
        return self._build_payload(
            loader_factory=loader_factory,
            input_sources=input_sources,
            input_backend_name=input_backend_name,
        )
