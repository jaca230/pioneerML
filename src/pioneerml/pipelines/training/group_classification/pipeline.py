from zenml import pipeline

from pioneerml.common.pipeline_utils.train import TrainingPipelineRunner
from .steps import (
    evaluate_group_classifier,
    export_group_classifier,
    train_group_classifier,
    tune_group_classifier,
)


_RUNNER = TrainingPipelineRunner()


@pipeline
def group_classification_pipeline(
    parquet_paths: list[str],
    pipeline_config: dict | None = None,
):
    return _RUNNER.run(
        pipeline_config=pipeline_config,
        load_dataset_fn=None,
        tune_fn=tune_group_classifier,
        train_fn=train_group_classifier,
        evaluate_fn=evaluate_group_classifier,
        export_fn=export_group_classifier,
        load_kwargs={"parquet_paths": parquet_paths},
    )
