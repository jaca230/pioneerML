from __future__ import annotations

from pioneerml.common.data_writer import WriterFactory

from .payloads import WriterFactoryInitStepPayload
from .resolvers.config import WriterRuntimeConfigResolver

from ..base_pipeline_step import BasePipelineStep


class BaseWriterFactoryInitStep(BasePipelineStep):
    DEFAULT_CONFIG = {
        "writer_name": None,
        "output_backend_name": "parquet",
        "fallback_output_dir": "data/inference",
        "output_dir": None,
        "output_path": None,
        "streaming": True,
        "write_timestamped": False,
        "timestamp": None,
        "writer_params": {},
    }
    config_resolver_classes = (WriterRuntimeConfigResolver,)
    payload_resolver_classes = ()

    def _build_payload(self, *, writer_factory: WriterFactory) -> WriterFactoryInitStepPayload:
        return WriterFactoryInitStepPayload(writer_factory=writer_factory)

    def _execute(
        self,
        *,
        output_dir: str | None = None,
        output_path: str | None = None,
    ) -> WriterFactoryInitStepPayload:
        writer_factory = self.runtime_state.get("writer_factory")
        if not isinstance(writer_factory, WriterFactory):
            raise RuntimeError(f"{self.__class__.__name__} runtime_state missing valid 'writer_factory'.")

        if output_dir is not None or output_path is not None:
            writer_factory = WriterRuntimeConfigResolver.build_writer_factory(
                cfg=dict(self.config_json),
                output_dir=output_dir,
                output_path=output_path,
            )
        return self._build_payload(writer_factory=writer_factory)
