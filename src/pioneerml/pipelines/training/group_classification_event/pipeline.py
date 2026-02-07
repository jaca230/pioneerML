from zenml import pipeline

from pioneerml.common.pipeline_utils.train import TrainingPipelineRunner
from .steps import (
    evaluate_group_classifier_event,
    export_group_classifier_event,
    load_group_classifier_event_dataset,
    train_group_classifier_event,
    tune_group_classifier_event,
)


_RUNNER = TrainingPipelineRunner()


@pipeline
def group_classification_event_pipeline(
    parquet_paths: list[str],
    pipeline_config: dict | None = None,
):
    return _RUNNER.run(
        pipeline_config=pipeline_config,
        load_dataset_fn=load_group_classifier_event_dataset,
        tune_fn=tune_group_classifier_event,
        train_fn=train_group_classifier_event,
        evaluate_fn=evaluate_group_classifier_event,
        export_fn=export_group_classifier_event,
        load_kwargs={"parquet_paths": parquet_paths},
    )
