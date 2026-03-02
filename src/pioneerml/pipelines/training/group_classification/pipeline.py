from __future__ import annotations

from zenml import pipeline

from pioneerml.common.pipeline.runners import TrainingPipelineRunner

from .steps import (
    evaluate_group_classifier_step,
    export_group_classifier_step,
    load_group_classifier_dataset_step,
    train_group_classifier_step,
    tune_group_classifier_step,
)

_RUNNER = TrainingPipelineRunner(
    load_step=load_group_classifier_dataset_step,
    hpo_step=tune_group_classifier_step,
    train_step=train_group_classifier_step,
    evaluate_step=evaluate_group_classifier_step,
    export_step=export_group_classifier_step,
)


@pipeline
def group_classification_pipeline(
    parquet_paths: list[str],
    pipeline_config: dict | None = None,
):
    return _RUNNER.run(
        loader_kwargs={"parquet_paths": parquet_paths},
        pipeline_config=pipeline_config,
    )

