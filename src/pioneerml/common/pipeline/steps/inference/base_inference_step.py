from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable
from typing import Any

import torch

from pioneerml.common.data_writer import BaseDataWriter, WriterFactory, WriterRunConfig
from pioneerml.common.data_writer.input_source import PredictionSet
from pioneerml.common.data_loader.config import DataFlowConfig
from pioneerml.common.data_loader.input_source import InputSourceSet

from ..base_pipeline_step import BasePipelineStep


@dataclass(frozen=True)
class InferenceSourceContext:
    source_idx: int
    src_path: Path
    num_rows: int
    source_event_offset: int


@dataclass(frozen=True)
class InferenceRuntime:
    cfg: dict
    device: torch.device
    model_path: str
    scripted: object
    writer: BaseDataWriter
    materialize_outputs: bool
    validated_files: list[str]
    validated_file_rows: list[int]
    flow_cfg: DataFlowConfig
    source_contexts: list[InferenceSourceContext]
    output_path: str | None


class BaseInferenceStep(BasePipelineStep):
    def execute(
        self,
        *,
        model_info: dict,
        inputs: dict,
        writer_setup: dict,
    ) -> dict:
        cfg = self.get_config()
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
        outputs = self.run_writer_inference(runtime=runtime, cfg=cfg, inputs=inputs, context=context)
        return self.finalize_inference_outputs(
            outputs=outputs,
            cfg=cfg,
            runtime=runtime,
            inputs=inputs,
            context=context,
        )

    def expected_writer_type(self) -> type[BaseDataWriter]:
        raise NotImplementedError(f"{self.__class__.__name__} must implement expected_writer_type().")

    def infer_prediction_sets_for_source(
        self,
        *,
        source_ctx: InferenceSourceContext,
        runtime: InferenceRuntime,
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
        runtime: InferenceRuntime,
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
        runtime: InferenceRuntime,
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
    def resolve_device(prefer_cuda: bool = True) -> torch.device:
        if prefer_cuda and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def load_torchscript(model_path: str, *, device: torch.device):
        scripted = torch.jit.load(model_path, map_location=device)
        scripted.eval()
        return scripted

    @staticmethod
    def make_writer_factory(*, writer_setup: dict) -> WriterFactory:
        resolved_output_dir = BaseDataWriter.ensure_output_dir(
            (None if writer_setup.get("output_dir") is None else str(writer_setup.get("output_dir"))),
            str(writer_setup.get("fallback_output_dir", "data/inference")),
        )
        run_config = WriterRunConfig(
            output_dir=resolved_output_dir,
            timestamp=(str(writer_setup["timestamp"]) if writer_setup.get("timestamp") is not None else BaseDataWriter.timestamp()),
            streaming=bool(writer_setup.get("streaming", True)),
            write_timestamped=bool(writer_setup.get("write_timestamped", False)),
        )
        return WriterFactory(
            writer_name=str(writer_setup["writer_name"]),
            output_backend_name=str(writer_setup.get("output_backend_name", "parquet")),
            run_config=run_config,
        )

    def setup_inference_runtime(
        self,
        *,
        cfg: dict,
        model_info: dict,
        inputs: dict,
        writer_setup: dict,
        prefer_cuda: bool = True,
        default_materialize_for_train_mode: bool = True,
    ) -> InferenceRuntime:
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
        return InferenceRuntime(
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
        explicit = cfg.get("materialize_outputs", None)
        if explicit is not None:
            return bool(explicit)
        if not default_for_train_mode:
            return False
        mode = str(inputs.get("mode", "inference")).strip().lower()
        return mode == "train"

    @staticmethod
    def resolve_validated_files(inputs: dict) -> list[str]:
        validated_files = [
            str(p)
            for p in (
                inputs.get("validated_files")
                or inputs.get("main_sources")
                or inputs.get("parquet_paths")
                or []
            )
        ]
        if not validated_files:
            raise RuntimeError("No validated files provided for inference.")
        return validated_files

    @staticmethod
    def resolve_validated_file_rows(inputs: dict, *, total_files: int) -> list[int]:
        rows = [int(v) for v in (inputs.get("validated_file_rows") or [])]
        if len(rows) != int(total_files):
            raise RuntimeError(
                f"Expected validated_file_rows aligned with validated_files ({total_files}), got {len(rows)}."
            )
        return rows

    @staticmethod
    def resolve_data_flow_config(
        inputs: dict,
        *,
        default_batch_size: int = 64,
        default_row_groups_per_chunk: int = 4,
        default_num_workers: int = 0,
    ) -> DataFlowConfig:
        flow_cfg_dict = dict(inputs.get("data_flow_config") or {})
        return DataFlowConfig(
            batch_size=int(flow_cfg_dict.get("batch_size", inputs.get("batch_size", default_batch_size))),
            row_groups_per_chunk=int(
                flow_cfg_dict.get("row_groups_per_chunk", inputs.get("row_groups_per_chunk", default_row_groups_per_chunk))
            ),
            num_workers=int(flow_cfg_dict.get("num_workers", inputs.get("num_workers", default_num_workers))),
        )

    @staticmethod
    def iter_source_contexts(*, validated_files: list[str], validated_file_rows: list[int]) -> list[InferenceSourceContext]:
        out: list[InferenceSourceContext] = []
        if len(validated_file_rows) != len(validated_files):
            raise RuntimeError(
                f"validated_file_rows must align with validated_files. Got {len(validated_file_rows)} vs {len(validated_files)}."
            )
        source_event_offset = 0
        for source_idx, (src_file, num_rows) in enumerate(zip(validated_files, validated_file_rows, strict=True)):
            src_path = Path(src_file).expanduser().resolve()
            out.append(
                InferenceSourceContext(
                    source_idx=int(source_idx),
                    src_path=src_path,
                    num_rows=num_rows,
                    source_event_offset=int(source_event_offset),
                )
            )
            source_event_offset += num_rows
        return out

    @staticmethod
    def select_source_aligned_paths(
        paths: list[str] | None,
        *,
        source_index: int,
        total_files: int,
        label: str,
    ) -> list[str] | None:
        if not paths:
            return None
        if len(paths) != int(total_files):
            raise RuntimeError(
                f"Expected {total_files} {label} files aligned with primary inputs, got {len(paths)}."
            )
        return [str(paths[source_index])]

    def resolve_optional_source_paths(
        self,
        *,
        inputs: dict,
        source_index: int,
        total_files: int,
        source_name: str,
        input_keys: tuple[str, ...],
    ) -> list[str] | None:
        source_paths: list[str] | None = None
        for key in input_keys:
            raw = inputs.get(key)
            if raw:
                source_paths = [str(p) for p in raw]
                break
        return self.select_source_aligned_paths(
            source_paths,
            source_index=source_index,
            total_files=total_files,
            label=source_name,
        )

    def build_input_source_for_path(
        self,
        *,
        src_path: Path,
        optional_sources: dict[str, list[str] | None] | None = None,
    ) -> InputSourceSet:
        dynamic_sources = {
            str(k): (list(v) if v is not None else None)
            for k, v in dict(optional_sources or {}).items()
        }
        return InputSourceSet(
            main_sources=[str(src_path)],
            optional_sources_by_name=dynamic_sources,
        )

    @staticmethod
    def ensure_writer_type(
        *,
        writer: BaseDataWriter,
        expected_type: type[BaseDataWriter],
    ) -> BaseDataWriter:
        if not isinstance(writer, expected_type):
            raise RuntimeError(
                f"Expected {expected_type.__name__}, got {type(writer).__name__}."
            )
        return writer

    def run_writer_inference(
        self,
        *,
        runtime: InferenceRuntime,
        cfg: dict,
        inputs: dict,
        context: dict[str, Any],
    ) -> dict:
        writer = runtime.writer
        writer_cfg = writer.run_config
        streaming = bool(writer_cfg.streaming)
        write_timestamped = bool(writer_cfg.write_timestamped)
        out_dir = writer_cfg.output_dir
        ts = writer_cfg.timestamp
        output_path = runtime.output_path
        source_contexts = runtime.source_contexts
        chunk_output_path = output_path if len(source_contexts) == 1 else None

        start_state = {
            "source_contexts": [{"src_path": str(c.src_path), "num_rows": int(c.num_rows)} for c in source_contexts],
            "output_dir": out_dir,
            "output_path": output_path,
            "write_timestamped": write_timestamped,
            "timestamp": ts,
            "streaming": streaming,
        }
        writer.on_start(state=start_state)

        total_rows = 0
        with torch.no_grad():
            for source_ctx in source_contexts:
                total_rows += int(source_ctx.num_rows)
                for prediction_set in self.infer_prediction_sets_for_source(
                    source_ctx=source_ctx,
                    runtime=runtime,
                    cfg=cfg,
                    inputs=inputs,
                    context=context,
                ):
                    writer.on_chunk(
                        state=writer.chunk_state(
                            prediction_set=prediction_set,
                            output_dir=out_dir,
                            output_path=chunk_output_path,
                            write_timestamped=write_timestamped,
                            timestamp=ts,
                        )
                    )

        finalized = writer.on_finalize(state=start_state)
        outputs = dict(finalized.get("run_outputs") or {})
        prediction_paths = [str(p) for p in outputs.get("predictions_paths") or []]
        timestamped_prediction_paths = [str(p) for p in outputs.get("timestamped_predictions_paths") or []]

        return {
            "predictions_path": prediction_paths[0] if len(prediction_paths) == 1 else None,
            "predictions_paths": prediction_paths,
            "timestamped_predictions_path": (
                timestamped_prediction_paths[0] if len(timestamped_prediction_paths) == 1 else None
            ),
            "timestamped_predictions_paths": timestamped_prediction_paths,
            "num_rows": int(total_rows),
            "validated_files": list(runtime.validated_files),
            "model_path": str(runtime.model_path),
            "streaming": bool(streaming),
        }
