from zenml import pipeline

from .steps import (
    evaluate_group_classifier_event,
    export_group_classifier_event,
    load_group_classifier_event_dataset,
    train_group_classifier_event,
    tune_group_classifier_event,
)


@pipeline
def group_classification_event_pipeline(
    parquet_paths: list[str],
    pipeline_config: dict | None = None,
):
    if pipeline_config is not None and not isinstance(pipeline_config, dict):
        raise TypeError(f"Expected dict for pipeline_config, got {type(pipeline_config).__name__}.")

    dataset = load_group_classifier_event_dataset(parquet_paths, pipeline_config=pipeline_config)
    hpo_params = tune_group_classifier_event(dataset, pipeline_config=pipeline_config)
    module = train_group_classifier_event(dataset, pipeline_config=pipeline_config, hpo_params=hpo_params)
    metrics = evaluate_group_classifier_event(module, dataset, pipeline_config=pipeline_config)
    export = export_group_classifier_event(
        module,
        dataset,
        pipeline_config=pipeline_config,
        hpo_params=hpo_params,
        metrics=metrics,
    )

    return module, dataset, metrics, export
