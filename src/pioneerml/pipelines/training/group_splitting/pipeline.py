from __future__ import annotations

from zenml import pipeline

from pioneerml.common.pipeline.runners import TrainingPipelineRunner

from .steps import (
    evaluate_group_splitter_step,
    export_group_splitter_step,
    load_group_splitter_dataset_step,
    train_group_splitter_step,
    tune_group_splitter_step,
)

_RUNNER = TrainingPipelineRunner(
    load_step=load_group_splitter_dataset_step,
    hpo_step=tune_group_splitter_step,
    train_step=train_group_splitter_step,
    evaluate_step=evaluate_group_splitter_step,
    export_step=export_group_splitter_step,
)


@pipeline
def group_splitting_pipeline(
    pipeline_config: dict | None = None,
):
    return _RUNNER.run(
        loader_kwargs={},
        pipeline_config=pipeline_config,
    )
