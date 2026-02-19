from zenml import step

from pioneerml.pipelines.inference.endpoint_regression.services import (
    EndpointRegressorInferenceInputsService,
)


@step(enable_cache=False)
def load_endpoint_regressor_inference_inputs(
    parquet_paths: list[str],
    group_probs_parquet_paths: list[str] | None = None,
    group_splitter_parquet_paths: list[str] | None = None,
    pipeline_config: dict | None = None,
) -> dict:
    service = EndpointRegressorInferenceInputsService(pipeline_config=pipeline_config)
    return service.execute(
        parquet_paths=parquet_paths,
        group_probs_parquet_paths=group_probs_parquet_paths,
        group_splitter_parquet_paths=group_splitter_parquet_paths,
    )
