from zenml import pipeline

from pioneerml.common.pipeline_utils.train import TrainingPipelineRunner
from .steps import (
    evaluate_event_splitter_event,
    export_event_splitter_event,
    load_event_splitter_event_dataset,
    train_event_splitter_event,
    tune_event_splitter_event,
)


_RUNNER = TrainingPipelineRunner()


@pipeline
def event_splitter_event_pipeline(
    parquet_paths: list[str],
    group_probs_parquet_paths: list[str] | None = None,
    group_splitter_parquet_paths: list[str] | None = None,
    endpoint_parquet_paths: list[str] | None = None,
    pipeline_config: dict | None = None,
):
    return _RUNNER.run(
        pipeline_config=pipeline_config,
        load_dataset_fn=load_event_splitter_event_dataset,
        tune_fn=tune_event_splitter_event,
        train_fn=train_event_splitter_event,
        evaluate_fn=evaluate_event_splitter_event,
        export_fn=export_event_splitter_event,
        load_kwargs={
            "parquet_paths": parquet_paths,
            "group_probs_parquet_paths": group_probs_parquet_paths,
            "group_splitter_parquet_paths": group_splitter_parquet_paths,
            "endpoint_parquet_paths": endpoint_parquet_paths,
        },
    )
