from zenml import pipeline

from .steps import (
    evaluate_event_splitter,
    export_event_splitter,
    load_event_splitter_dataset,
    train_event_splitter,
    tune_event_splitter,
)


@pipeline
def event_splitting_pipeline(
    parquet_paths: list[str],
    group_probs_parquet_paths: list[str] | None = None,
    group_splitter_parquet_paths: list[str] | None = None,
    endpoint_parquet_paths: list[str] | None = None,
    pipeline_config: dict | None = None,
):
    dataset = load_event_splitter_dataset(
        parquet_paths=parquet_paths,
        group_probs_parquet_paths=group_probs_parquet_paths,
        group_splitter_parquet_paths=group_splitter_parquet_paths,
        endpoint_parquet_paths=endpoint_parquet_paths,
        pipeline_config=pipeline_config,
    )
    hpo_params = tune_event_splitter(dataset, pipeline_config=pipeline_config)
    module = train_event_splitter(dataset, pipeline_config=pipeline_config, hpo_params=hpo_params)
    metrics = evaluate_event_splitter(module, dataset, pipeline_config=pipeline_config)
    export = export_event_splitter(
        module,
        dataset,
        pipeline_config=pipeline_config,
        hpo_params=hpo_params,
        metrics=metrics,
    )
    return module, dataset, metrics, export
