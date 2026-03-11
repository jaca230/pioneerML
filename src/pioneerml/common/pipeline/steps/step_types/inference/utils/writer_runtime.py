from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable

import torch

from pioneerml.common.data_writer import BaseDataWriter, WriterFactory, WriterRunConfig
from pioneerml.common.data_writer.input_source import PredictionSet
from pioneerml.common.pipeline.payloads import InferenceRuntimePayload


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
        writer_params=dict(writer_setup.get("writer_params") or {}),
    )


def ensure_writer_type(*, writer: BaseDataWriter, expected_type: type[BaseDataWriter]) -> BaseDataWriter:
    if not isinstance(writer, expected_type):
        raise RuntimeError(f"Expected {expected_type.__name__}, got {type(writer).__name__}.")
    return writer


def run_writer_inference(
    *,
    runtime: InferenceRuntimePayload,
    infer_prediction_sets_for_source: Callable[..., Iterable[PredictionSet]],
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
            for prediction_set in infer_prediction_sets_for_source(
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
    prediction_paths = [str(path) for path in outputs.get("predictions_paths") or []]
    timestamped_prediction_paths = [str(path) for path in outputs.get("timestamped_predictions_paths") or []]

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
