from zenml import pipeline

from .steps import (
    evaluate_positron_angle,
    export_positron_angle,
    load_positron_angle_dataset,
    train_positron_angle,
    tune_positron_angle,
)


@pipeline
def positron_angle_regression_pipeline(
    parquet_paths: list[str],
    group_probs_parquet_paths: list[str] | None = None,
    group_splitter_parquet_paths: list[str] | None = None,
    endpoint_parquet_paths: list[str] | None = None,
    event_splitter_parquet_paths: list[str] | None = None,
    pion_stop_parquet_paths: list[str] | None = None,
    pipeline_config: dict | None = None,
):
    dataset = load_positron_angle_dataset(
        parquet_paths=parquet_paths,
        group_probs_parquet_paths=group_probs_parquet_paths,
        group_splitter_parquet_paths=group_splitter_parquet_paths,
        endpoint_parquet_paths=endpoint_parquet_paths,
        event_splitter_parquet_paths=event_splitter_parquet_paths,
        pion_stop_parquet_paths=pion_stop_parquet_paths,
        pipeline_config=pipeline_config,
    )
    hpo_params = tune_positron_angle(dataset, pipeline_config=pipeline_config)
    module = train_positron_angle(dataset, pipeline_config=pipeline_config, hpo_params=hpo_params)
    metrics = evaluate_positron_angle(module, dataset, pipeline_config=pipeline_config)
    export = export_positron_angle(
        module,
        dataset,
        pipeline_config=pipeline_config,
        hpo_params=hpo_params,
        metrics=metrics,
    )
    return module, dataset, metrics, export
