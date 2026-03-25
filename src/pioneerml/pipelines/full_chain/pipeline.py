import gc
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from zenml import pipeline, step

from pioneerml.common.pipeline.steps import (
    BaseEvaluationStep,
    BaseExportStep,
    BaseFullTrainingStep,
    BaseHPOStep,
    BaseInferenceStep,
    BaseModelHandleBuilderStep,
)

MODEL_CHAIN: tuple[str, ...] = (
    "group_classifier",
    "group_splitter",
    "endpoint_regression",
    "event_splitter",
    "pion_stop",
    "positron_angle",
)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any] | None) -> dict[str, Any]:
    out = deepcopy(dict(base))
    if override is None:
        return out
    for key, value in dict(override).items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[str(key)] = _deep_merge(out[key], value)
        else:
            out[str(key)] = deepcopy(value)
    return out


def _resolve_stage_config(
    *,
    pipeline_config: Mapping[str, Any] | None,
    model_key: str,
    stage_key: str,
) -> dict[str, Any]:
    cfg = dict(pipeline_config or {})

    common_cfg = cfg.get("common")
    stage_common: dict[str, Any] = {}
    if isinstance(common_cfg, Mapping):
        raw = common_cfg.get(stage_key)
        if isinstance(raw, Mapping):
            stage_common = dict(raw)

    model_cfg = cfg.get(model_key)
    if not isinstance(model_cfg, Mapping):
        raise KeyError(
            f"Missing top-level model config '{model_key}'. "
            f"Expected keys: {list(MODEL_CHAIN)} and optional 'common'."
        )

    stage_cfg = model_cfg.get(stage_key)
    if not isinstance(stage_cfg, Mapping):
        raise KeyError(
            f"Missing '{stage_key}' config for model '{model_key}'. "
            "Expected shape: {model_key: {'training': {...}, 'inference': {...}}}"
        )

    return _deep_merge(stage_common, stage_cfg)


def _inject_model_path_from_export(
    *,
    inference_stage_config: Mapping[str, Any],
    training_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    cfg = _deep_merge({}, inference_stage_config)
    if not isinstance(training_payload, Mapping):
        return cfg

    export_payload = training_payload.get("export")
    if not isinstance(export_payload, Mapping):
        return cfg

    torchscript_path = export_payload.get("torchscript_path")
    if not isinstance(torchscript_path, str) or torchscript_path.strip() == "":
        return cfg

    model_handle_builder = cfg.get("model_handle_builder")
    if not isinstance(model_handle_builder, Mapping):
        return cfg

    model_handle = model_handle_builder.get("model_handle")
    if not isinstance(model_handle, Mapping):
        return cfg

    model_cfg = model_handle.get("config")
    if not isinstance(model_cfg, Mapping):
        return cfg

    current_model_path = model_cfg.get("model_path")
    if current_model_path in (None, "", "none", "None"):
        out = _deep_merge({}, cfg)
        out["model_handle_builder"]["model_handle"]["config"]["model_path"] = str(torchscript_path)
        return out

    return cfg


class ChainHPOStep(BaseHPOStep):
    step_key = "hpo"


class ChainTrainingStep(BaseFullTrainingStep):
    step_key = "train"


class ChainEvaluationStep(BaseEvaluationStep):
    step_key = "evaluate"


class ChainExportStep(BaseExportStep):
    step_key = "export"


class ChainModelHandleBuilderStep(BaseModelHandleBuilderStep):
    step_key = "model_handle_builder"


class ChainInferenceStep(BaseInferenceStep):
    step_key = "inference"


@step(enable_cache=False)
def run_training_stage_step(
    stage_key: str,
    stage_config: dict | None = None,
    depends_on: Any | None = None,
) -> Any:
    _ = depends_on
    cfg = dict(stage_config or {})

    hpo_payload = ChainHPOStep(pipeline_config=cfg).execute()
    train_payload = ChainTrainingStep(pipeline_config=cfg).execute(
        payloads={"hpo": hpo_payload},
    )
    metrics_payload = ChainEvaluationStep(pipeline_config=cfg).execute(
        payloads={"train": train_payload},
    )
    export_payload = ChainExportStep(pipeline_config=cfg).execute(
        payloads={
            "train": train_payload,
            "hpo": hpo_payload,
            "metrics": metrics_payload,
        },
    )

    return {
        "stage_key": str(stage_key),
        "hpo": hpo_payload,
        "train": train_payload,
        "metrics": metrics_payload,
        "export": export_payload,
    }


@step(enable_cache=False)
def run_inference_stage_step(
    stage_key: str,
    stage_config: dict | None = None,
    training_payload: dict | None = None,
    depends_on: Any | None = None,
) -> Any:
    _ = depends_on
    cfg = _inject_model_path_from_export(
        inference_stage_config=dict(stage_config or {}),
        training_payload=(None if training_payload is None else dict(training_payload)),
    )

    model_handle_payload = ChainModelHandleBuilderStep(pipeline_config=cfg).execute()
    inference_payload = ChainInferenceStep(pipeline_config=cfg).execute(
        payloads={"model_handle_builder": model_handle_payload},
    )

    return {
        "stage_key": str(stage_key),
        "model_handle": model_handle_payload,
        "inference": inference_payload,
    }


@step(enable_cache=False)
def run_cleanup_step(
    tag: str,
    depends_on: Any | None = None,
) -> Any:
    _ = depends_on
    out: dict[str, Any] = {
        "tag": str(tag),
        "gc_collected": int(gc.collect()),
    }

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            out["cuda_cache_cleared"] = True
        else:
            out["cuda_cache_cleared"] = False
    except Exception:
        out["cuda_cache_cleared"] = False

    try:
        import pyarrow as pa

        pa.default_memory_pool().release_unused()
        out["pyarrow_memory_released"] = True
    except Exception:
        out["pyarrow_memory_released"] = False

    return out


@pipeline
def full_chain_pipeline(
    pipeline_config: dict | None = None,
):
    ordered_stage_outputs: list[Any] = []
    upstream_dependency: Any | None = None

    for model_key in MODEL_CHAIN:
        training_cfg = _resolve_stage_config(
            pipeline_config=pipeline_config,
            model_key=model_key,
            stage_key="training",
        )
        inference_cfg = _resolve_stage_config(
            pipeline_config=pipeline_config,
            model_key=model_key,
            stage_key="inference",
        )

        training_output = run_training_stage_step(
            stage_key=model_key,
            stage_config=training_cfg,
            depends_on=upstream_dependency,
        )
        ordered_stage_outputs.append(training_output)

        after_training_gc = run_cleanup_step(
            tag=f"{model_key}_after_training",
            depends_on=training_output,
        )

        inference_output = run_inference_stage_step(
            stage_key=model_key,
            stage_config=inference_cfg,
            training_payload=training_output,
            depends_on=after_training_gc,
        )
        ordered_stage_outputs.append(inference_output)

        upstream_dependency = run_cleanup_step(
            tag=f"{model_key}_after_inference",
            depends_on=inference_output,
        )

    return tuple(ordered_stage_outputs)
