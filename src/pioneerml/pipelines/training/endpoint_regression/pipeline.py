from __future__ import annotations

from zenml import pipeline

from pioneerml.common.pipeline.runners import TrainingPipelineRunner

from .steps import (
    evaluate_endpoint_regressor_step,
    export_endpoint_regressor_step,
    load_endpoint_regressor_dataset_step,
    train_endpoint_regressor_step,
    tune_endpoint_regressor_step,
)

_RUNNER = TrainingPipelineRunner(
    load_step=load_endpoint_regressor_dataset_step,
    hpo_step=tune_endpoint_regressor_step,
    train_step=train_endpoint_regressor_step,
    evaluate_step=evaluate_endpoint_regressor_step,
    export_step=export_endpoint_regressor_step,
)


@pipeline
def endpoint_regression_pipeline(
    parquet_input_set: dict,
    pipeline_config: dict | None = None,
):
    return _RUNNER.run(
        loader_kwargs={"parquet_input_set": parquet_input_set},
        pipeline_config=pipeline_config,
    )
