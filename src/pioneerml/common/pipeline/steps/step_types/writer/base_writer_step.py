from __future__ import annotations

from dataclasses import asdict

from .payloads import WriterStepPayload
from .resolvers.base_writer_config_resolver import BaseWriterConfigResolver
from .resolvers.writer_setup_resolver import WriterSetupResolver

from ..base_pipeline_step import BasePipelineStep


class BaseWriterStep(BasePipelineStep):
    writer_name: str | None = None
    resolver_classes = (BaseWriterConfigResolver, WriterSetupResolver)
    DEFAULT_CONFIG = {"config_json": {}}

    def resolve_writer_setup(
        self,
        *,
        output_dir: str | None,
        output_path: str | None,
    ) -> dict:
        cfg = {"config_json": dict(self.config_json.get("writer_setup_config") or {})}
        writer_name = self.writer_name
        if writer_name is None:
            raise RuntimeError(f"{self.__class__.__name__} must define writer_name.")
        setup = WriterSetupResolver.resolve_writer_setup(
            cfg=cfg,
            default_writer_name=writer_name,
            output_dir=output_dir,
            output_path=output_path,
        )
        return asdict(setup)

    def build_payload(self, *, writer_setup: dict, inputs: dict) -> WriterStepPayload:
        return WriterStepPayload(
            writer_setup=dict(writer_setup),
            inputs=dict(inputs),
        )

    def _execute(
        self,
        *,
        inputs: dict,
        output_dir: str | None = None,
        output_path: str | None = None,
    ) -> WriterStepPayload:
        writer_setup = self.resolve_writer_setup(output_dir=output_dir, output_path=output_path)
        return self.build_payload(writer_setup=writer_setup, inputs=inputs)
