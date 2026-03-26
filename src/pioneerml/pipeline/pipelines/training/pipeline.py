from typing import Any

from zenml import pipeline, step

from pioneerml.pipeline.steps import (
    BaseEvaluationStep,
    BaseExportStep,
    BaseFullTrainingStep,
    BaseHPOStep,
)


class UnifiedHPOStep(BaseHPOStep):
    step_key = "hpo"


class UnifiedTrainingStep(BaseFullTrainingStep):
    step_key = "train"


class UnifiedEvaluationStep(BaseEvaluationStep):
    step_key = "evaluate"


class UnifiedExportStep(BaseExportStep):
    step_key = "export"


@step(name="tune_model", enable_cache=False)
def tune_model_step(
    pipeline_config: dict | None = None,
) -> Any:
    return UnifiedHPOStep(pipeline_config=pipeline_config).execute()


@step(name="train_model", enable_cache=False)
def train_model_step(
    hpo_payload: dict | None = None,
    pipeline_config: dict | None = None,
) -> Any:
    return UnifiedTrainingStep(pipeline_config=pipeline_config).execute(
        payloads={"hpo": hpo_payload},
    )


@step(name="evaluate_model", enable_cache=False)
def evaluate_model_step(
    train_payload: dict,
    pipeline_config: dict | None = None,
) -> Any:
    return UnifiedEvaluationStep(pipeline_config=pipeline_config).execute(
        payloads={"train": train_payload},
    )


@step(name="export_model", enable_cache=False)
def export_model_step(
    train_payload: dict,
    hpo_payload: dict | None = None,
    metrics: dict | None = None,
    pipeline_config: dict | None = None,
) -> Any:
    return UnifiedExportStep(pipeline_config=pipeline_config).execute(
        payloads={
            "train": train_payload,
            "hpo": hpo_payload,
            "metrics": metrics,
        }
    )


@pipeline
def training_pipeline(
    pipeline_config: dict | None = None,
):
    hpo_output = tune_model_step(pipeline_config=pipeline_config)
    train_output = train_model_step(hpo_payload=hpo_output, pipeline_config=pipeline_config)
    metrics = evaluate_model_step(train_payload=train_output, pipeline_config=pipeline_config)
    export = export_model_step(
        train_payload=train_output,
        hpo_payload=hpo_output,
        metrics=metrics,
        pipeline_config=pipeline_config,
    )
    return train_output, hpo_output, metrics, export
