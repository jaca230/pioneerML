from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pioneerml.common.data_loader.input_source import InputSourceSet
from pioneerml.common.data_writer import BaseDataWriter
from pioneerml.common.data_writer.input_source import PredictionSet
from pioneerml.common.pipeline.payloads import InferenceRuntimePayload, InferenceSourcePayload
from .payloads import InferenceStepPayload
from .resolvers.runtime_resolver import (
    InferenceRuntimeResolver,
)
from .utils.writer_runtime import (
    ensure_writer_type,
    make_writer_factory,
    run_writer_inference,
)

from ..base_pipeline_step import BasePipelineStep


class BaseInferenceStep(BasePipelineStep):
    resolver_classes = (InferenceRuntimeResolver,)
    DEFAULT_CONFIG = {"threshold": 0.5, "materialize_outputs": None}

    def _execute(
        self,
        *,
        model_info: dict,
        inputs: dict,
        writer_setup: dict,
    ) -> dict:
        cfg = dict(self.config_json.get("inference_config") or {})
        runtime = self.setup_inference_runtime(
            cfg=cfg,
            model_info=model_info,
            inputs=inputs,
            writer_setup=writer_setup,
            prefer_cuda=self.prefer_cuda_for_inference(cfg=cfg, inputs=inputs),
            default_materialize_for_train_mode=self.default_materialize_for_train_mode(cfg=cfg, inputs=inputs),
        )
        self.ensure_writer_type(writer=runtime.writer, expected_type=self.expected_writer_type())
        context = self.prepare_inference_context(cfg=cfg, runtime=runtime, inputs=inputs)
        outputs = run_writer_inference(
            runtime=runtime,
            infer_prediction_sets_for_source=self.infer_prediction_sets_for_source,
            cfg=cfg,
            inputs=inputs,
            context=context,
        )
        out = self.finalize_inference_outputs(
            outputs=outputs,
            cfg=cfg,
            runtime=runtime,
            inputs=inputs,
            context=context,
        )
        return self.build_payload(outputs=out)

    def expected_writer_type(self) -> type[BaseDataWriter]:
        raise NotImplementedError(f"{self.__class__.__name__} must implement expected_writer_type().")

    def infer_prediction_sets_for_source(
        self,
        *,
        source_ctx: InferenceSourcePayload,
        runtime: InferenceRuntimePayload,
        cfg: dict,
        inputs: dict,
        context: dict[str, Any],
    ) -> Iterable[PredictionSet]:
        _ = source_ctx
        _ = runtime
        _ = cfg
        _ = inputs
        _ = context
        raise NotImplementedError(f"{self.__class__.__name__} must implement infer_prediction_sets_for_source(...).")

    def prepare_inference_context(
        self,
        *,
        cfg: dict,
        runtime: InferenceRuntimePayload,
        inputs: dict,
    ) -> dict[str, Any]:
        _ = cfg
        _ = runtime
        _ = inputs
        return {}

    def finalize_inference_outputs(
        self,
        *,
        outputs: dict,
        cfg: dict,
        runtime: InferenceRuntimePayload,
        inputs: dict,
        context: dict[str, Any],
    ) -> dict:
        _ = cfg
        _ = runtime
        _ = inputs
        _ = context
        return outputs

    def prefer_cuda_for_inference(self, *, cfg: dict, inputs: dict) -> bool:
        _ = cfg
        _ = inputs
        return True

    def default_materialize_for_train_mode(self, *, cfg: dict, inputs: dict) -> bool:
        _ = cfg
        _ = inputs
        return True

    @staticmethod
    def resolve_device(prefer_cuda: bool = True):
        return InferenceRuntimeResolver.resolve_device(prefer_cuda=prefer_cuda)

    @staticmethod
    def load_torchscript(model_path: str, *, device):
        return InferenceRuntimeResolver.load_torchscript(model_path=model_path, device=device)

    @staticmethod
    def make_writer_factory(*, writer_setup: dict):
        return make_writer_factory(writer_setup=writer_setup)

    def setup_inference_runtime(
        self,
        *,
        cfg: dict,
        model_info: dict,
        inputs: dict,
        writer_setup: dict,
        prefer_cuda: bool = True,
        default_materialize_for_train_mode: bool = True,
    ) -> InferenceRuntimePayload:
        writer_factory = self.make_writer_factory(writer_setup=writer_setup)
        writer = writer_factory.create()
        streaming = bool(writer.run_config.streaming)
        materialize_outputs = self.resolve_materialize_outputs(
            cfg,
            inputs=inputs,
            default_for_train_mode=default_materialize_for_train_mode,
        )
        if not streaming:
            materialize_outputs = True
        device = self.resolve_device(prefer_cuda=prefer_cuda)

        model_path = str(model_info["model_path"])
        scripted = self.load_torchscript(model_path=model_path, device=device)

        validated_files = self.resolve_validated_files(inputs)
        validated_file_rows = self.resolve_validated_file_rows(inputs, total_files=len(validated_files))
        flow_cfg = self.resolve_data_flow_config(inputs)
        source_contexts = self.iter_source_contexts(
            validated_files=validated_files,
            validated_file_rows=validated_file_rows,
        )
        return InferenceRuntimePayload(
            cfg=dict(cfg),
            device=device,
            model_path=model_path,
            scripted=scripted,
            writer=writer,
            materialize_outputs=bool(materialize_outputs),
            validated_files=validated_files,
            validated_file_rows=validated_file_rows,
            flow_cfg=flow_cfg,
            source_contexts=source_contexts,
            output_path=(None if writer_setup.get("output_path") is None else str(writer_setup.get("output_path"))),
        )

    @staticmethod
    def resolve_streaming_flag(cfg: dict, *, default: bool = True) -> bool:
        return bool(cfg.get("streaming", default))

    @staticmethod
    def resolve_materialize_outputs(
        cfg: dict,
        *,
        inputs: dict,
        default_for_train_mode: bool = True,
    ) -> bool:
        return InferenceRuntimeResolver.resolve_materialize_outputs(
            cfg,
            inputs=inputs,
            default_for_train_mode=default_for_train_mode,
        )

    @staticmethod
    def resolve_validated_files(inputs: dict) -> list[str]:
        return InferenceRuntimeResolver.resolve_validated_files(inputs)

    @staticmethod
    def resolve_validated_file_rows(inputs: dict, *, total_files: int) -> list[int]:
        return InferenceRuntimeResolver.resolve_validated_file_rows(inputs, total_files=total_files)

    @staticmethod
    def resolve_data_flow_config(
        inputs: dict,
        *,
        default_batch_size: int = 64,
        default_row_groups_per_chunk: int = 4,
        default_num_workers: int = 0,
    ):
        return InferenceRuntimeResolver.resolve_data_flow_config(
            inputs,
            default_batch_size=default_batch_size,
            default_row_groups_per_chunk=default_row_groups_per_chunk,
            default_num_workers=default_num_workers,
        )

    @staticmethod
    def iter_source_contexts(*, validated_files: list[str], validated_file_rows: list[int]) -> list[InferenceSourcePayload]:
        return InferenceRuntimeResolver.iter_source_contexts(
            validated_files=validated_files,
            validated_file_rows=validated_file_rows,
        )

    @staticmethod
    def select_source_aligned_paths(
        paths: list[str] | None,
        *,
        source_index: int,
        total_files: int,
        label: str,
    ) -> list[str] | None:
        return InferenceRuntimeResolver.select_source_aligned_paths(
            paths,
            source_index=source_index,
            total_files=total_files,
            label=label,
        )

    def resolve_optional_source_paths(
        self,
        *,
        inputs: dict,
        source_index: int,
        total_files: int,
        source_name: str,
        input_keys: tuple[str, ...],
    ) -> list[str] | None:
        return InferenceRuntimeResolver.resolve_optional_source_paths(
            inputs=inputs,
            source_index=source_index,
            total_files=total_files,
            source_name=source_name,
            input_keys=input_keys,
        )

    def build_input_source_for_path(
        self,
        *,
        src_path: Path,
        optional_sources: dict[str, list[str] | None] | None = None,
    ) -> InputSourceSet:
        return InferenceRuntimeResolver.build_input_source_for_path(
            src_path=src_path,
            optional_sources=optional_sources,
        )

    @staticmethod
    def ensure_writer_type(
        *,
        writer: BaseDataWriter,
        expected_type: type[BaseDataWriter],
    ) -> BaseDataWriter:
        return ensure_writer_type(writer=writer, expected_type=expected_type)

    def build_payload(self, *, outputs: dict) -> InferenceStepPayload:
        return InferenceStepPayload(**dict(outputs))
