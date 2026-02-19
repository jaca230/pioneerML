from zenml import pipeline

from .steps import (
    evaluate_pion_stop,
    export_pion_stop,
    load_pion_stop_dataset,
    train_pion_stop,
    tune_pion_stop,
)


@pipeline
def pion_stop_regression_pipeline(
    parquet_paths: list[str],
    group_probs_parquet_paths: list[str] | None = None,
    group_splitter_parquet_paths: list[str] | None = None,
    endpoint_parquet_paths: list[str] | None = None,
    event_splitter_parquet_paths: list[str] | None = None,
    pipeline_config: dict | None = None,
):
    dataset = load_pion_stop_dataset(
        parquet_paths=parquet_paths,
        group_probs_parquet_paths=group_probs_parquet_paths,
        group_splitter_parquet_paths=group_splitter_parquet_paths,
        endpoint_parquet_paths=endpoint_parquet_paths,
        event_splitter_parquet_paths=event_splitter_parquet_paths,
        pipeline_config=pipeline_config,
    )
    hpo_params = tune_pion_stop(dataset, pipeline_config=pipeline_config)
    module = train_pion_stop(dataset, pipeline_config=pipeline_config, hpo_params=hpo_params)
    metrics = evaluate_pion_stop(module, dataset, pipeline_config=pipeline_config)
    export = export_pion_stop(
        module,
        dataset,
        pipeline_config=pipeline_config,
        hpo_params=hpo_params,
        metrics=metrics,
    )
    return module, dataset, metrics, export
