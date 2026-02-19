from zenml import step

from pioneerml.pipelines.inference.event_splitting.services import EventSplitterInferenceInputsService


@step(enable_cache=False)
def load_event_splitter_inference_inputs(
    parquet_paths: list[str],
    group_probs_parquet_paths: list[str] | None = None,
    group_splitter_parquet_paths: list[str] | None = None,
    endpoint_parquet_paths: list[str] | None = None,
    pipeline_config: dict | None = None,
) -> dict:
    service = EventSplitterInferenceInputsService(pipeline_config=pipeline_config)
    return service.execute(
        parquet_paths=parquet_paths,
        group_probs_parquet_paths=group_probs_parquet_paths,
        group_splitter_parquet_paths=group_splitter_parquet_paths,
        endpoint_parquet_paths=endpoint_parquet_paths,
    )
