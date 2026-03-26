from typing import Any

from zenml import pipeline, step

from pioneerml.pipeline.steps import (
    BaseInferenceStep,
    BaseModelHandleBuilderStep,
)


class UnifiedModelHandleBuilderStep(BaseModelHandleBuilderStep):
    step_key = "model_handle_builder"


class UnifiedInferenceStep(BaseInferenceStep):
    step_key = "inference"


@step(name="build_model_handle", enable_cache=False)
def build_model_handle_step(
    pipeline_config: dict | None = None,
) -> Any:
    return UnifiedModelHandleBuilderStep(pipeline_config=pipeline_config).execute()


@step(name="run_inference", enable_cache=False)
def run_inference_step(
    model_handle_payload: dict,
    pipeline_config: dict | None = None,
) -> Any:
    return UnifiedInferenceStep(pipeline_config=pipeline_config).execute(
        payloads={"model_handle_builder": model_handle_payload},
    )


@pipeline
def inference_pipeline(
    pipeline_config: dict | None = None,
):
    model_handle_payload = build_model_handle_step(
        pipeline_config=pipeline_config,
    )
    inference_output = run_inference_step(
        model_handle_payload=model_handle_payload,
        pipeline_config=pipeline_config,
    )
    return inference_output
