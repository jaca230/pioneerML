from __future__ import annotations

from zenml import pipeline

from pioneerml.common.pipeline.runners import InferencePipelineRunner

from .steps import (
    load_endpoint_regressor_inference_inputs_step,
    load_endpoint_regressor_model_step,
    run_endpoint_regressor_inference_step,
    save_endpoint_regressor_predictions_step,
)

_RUNNER = InferencePipelineRunner(
    load_inputs_step=load_endpoint_regressor_inference_inputs_step,
    load_model_step=load_endpoint_regressor_model_step,
    run_inference_step=run_endpoint_regressor_inference_step,
    save_predictions_step=save_endpoint_regressor_predictions_step,
)


@pipeline
def endpoint_regression_inference_pipeline(
    parquet_input_set: dict,
    model_path: str | None = None,
    output_dir: str | None = None,
    output_path: str | None = None,
    pipeline_config: dict | None = None,
):
    return _RUNNER.run(
        loader_kwargs={"parquet_input_set": parquet_input_set},
        model_path=model_path,
        output_dir=output_dir,
        output_path=output_path,
        pipeline_config=pipeline_config,
    )
