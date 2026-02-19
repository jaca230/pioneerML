from zenml import pipeline

from .steps import (
    evaluate_endpoint_regressor,
    export_endpoint_regressor,
    load_endpoint_regressor_dataset,
    train_endpoint_regressor,
    tune_endpoint_regressor,
)


@pipeline
def endpoint_regression_pipeline(
    parquet_paths: list[str],
    group_probs_parquet_paths: list[str] | None = None,
    group_splitter_parquet_paths: list[str] | None = None,
    pipeline_config: dict | None = None,
):
    dataset = load_endpoint_regressor_dataset(
        parquet_paths=parquet_paths,
        group_probs_parquet_paths=group_probs_parquet_paths,
        group_splitter_parquet_paths=group_splitter_parquet_paths,
        pipeline_config=pipeline_config,
    )
    hpo_params = tune_endpoint_regressor(dataset, pipeline_config=pipeline_config)
    module = train_endpoint_regressor(dataset, pipeline_config=pipeline_config, hpo_params=hpo_params)
    metrics = evaluate_endpoint_regressor(module, dataset, pipeline_config=pipeline_config)
    export = export_endpoint_regressor(
        module,
        dataset,
        pipeline_config=pipeline_config,
        hpo_params=hpo_params,
        metrics=metrics,
    )
    return module, dataset, metrics, export
