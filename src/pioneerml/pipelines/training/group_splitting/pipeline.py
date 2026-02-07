from zenml import pipeline

from pioneerml.common.pipeline_utils.train import TrainingPipelineRunner
from .steps import (
    evaluate_group_splitter,
    export_group_splitter,
    load_group_splitter_dataset,
    train_group_splitter,
    tune_group_splitter,
)


_RUNNER = TrainingPipelineRunner()


@pipeline
def group_splitting_pipeline(
    parquet_paths: list[str],
    group_probs_parquet_paths: list[str] | None = None,
    pipeline_config: dict | None = None,
):
    return _RUNNER.run(
        pipeline_config=pipeline_config,
        load_dataset_fn=load_group_splitter_dataset,
        tune_fn=tune_group_splitter,
        train_fn=train_group_splitter,
        evaluate_fn=evaluate_group_splitter,
        export_fn=export_group_splitter,
        load_kwargs={
            "parquet_paths": parquet_paths,
            "group_probs_parquet_paths": group_probs_parquet_paths,
        },
    )
