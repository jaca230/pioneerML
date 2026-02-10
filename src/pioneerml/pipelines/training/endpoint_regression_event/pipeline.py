from zenml import pipeline

from pioneerml.common.pipeline_utils.train import TrainingPipelineRunner
from .steps import (
    evaluate_endpoint_regressor_event,
    export_endpoint_regressor_event,
    load_endpoint_regressor_event_dataset,
    train_endpoint_regressor_event,
    tune_endpoint_regressor_event,
)


_RUNNER = TrainingPipelineRunner()


@pipeline
def endpoint_regression_event_pipeline(
    parquet_paths: list[str],
    group_probs_parquet_paths: list[str] | None = None,
    group_splitter_parquet_paths: list[str] | None = None,
    pipeline_config: dict | None = None,
):
    return _RUNNER.run(
        pipeline_config=pipeline_config,
        load_dataset_fn=load_endpoint_regressor_event_dataset,
        tune_fn=tune_endpoint_regressor_event,
        train_fn=train_endpoint_regressor_event,
        evaluate_fn=evaluate_endpoint_regressor_event,
        export_fn=export_endpoint_regressor_event,
        load_kwargs={
            "parquet_paths": parquet_paths,
            "group_probs_parquet_paths": group_probs_parquet_paths,
            "group_splitter_parquet_paths": group_splitter_parquet_paths,
        },
    )
