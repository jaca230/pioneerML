from zenml import step

from pioneerml.pipelines.inference.pion_stop.services import (
    PionStopInferenceInputsService,
)


@step(enable_cache=False)
def load_pion_stop_inference_inputs(
    parquet_paths: list[str],
    group_probs_parquet_paths: list[str] | None = None,
    group_splitter_parquet_paths: list[str] | None = None,
    endpoint_parquet_paths: list[str] | None = None,
    event_splitter_parquet_paths: list[str] | None = None,
    pipeline_config: dict | None = None,
) -> dict:
    service = PionStopInferenceInputsService(pipeline_config=pipeline_config)
    return service.execute(
        parquet_paths=parquet_paths,
        group_probs_parquet_paths=group_probs_parquet_paths,
        group_splitter_parquet_paths=group_splitter_parquet_paths,
        endpoint_parquet_paths=endpoint_parquet_paths,
        event_splitter_parquet_paths=event_splitter_parquet_paths,
    )
