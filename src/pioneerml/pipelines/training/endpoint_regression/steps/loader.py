from zenml import step

from pioneerml.pipelines.training.endpoint_regression.dataset import EndpointRegressorDataset


@step
def load_endpoint_regressor_dataset(
    parquet_paths: list[str],
    group_probs_parquet_paths: list[str] | None = None,
    group_splitter_parquet_paths: list[str] | None = None,
    pipeline_config: dict | None = None,
) -> EndpointRegressorDataset:
    raise NotImplementedError(
        "load_endpoint_regressor_dataset was moved to deprecated for C++-loader retirement. "
        "Implement the new pure-Python/PyG loader for this pipeline."
    )
