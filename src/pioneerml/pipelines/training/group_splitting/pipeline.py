from zenml import pipeline

from .steps import (
    evaluate_group_splitter,
    export_group_splitter,
    load_group_splitter_dataset,
    train_group_splitter,
    tune_group_splitter,
)


@pipeline
def group_splitting_pipeline(
    parquet_paths: list[str],
    group_probs_parquet_paths: list[str] | None = None,
    pipeline_config: dict | None = None,
):
    dataset = load_group_splitter_dataset(
        parquet_paths=parquet_paths,
        group_probs_parquet_paths=group_probs_parquet_paths,
        pipeline_config=pipeline_config,
    )
    hpo_params = tune_group_splitter(dataset, pipeline_config=pipeline_config)
    module = train_group_splitter(dataset, pipeline_config=pipeline_config, hpo_params=hpo_params)
    metrics = evaluate_group_splitter(module, dataset, pipeline_config=pipeline_config)
    export = export_group_splitter(
        module,
        dataset,
        pipeline_config=pipeline_config,
        hpo_params=hpo_params,
        metrics=metrics,
    )
    return module, dataset, metrics, export
