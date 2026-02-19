from zenml import step

from pioneerml.pipelines.inference.positron_angle.services import (
    PositronAngleInferenceInputsService,
)


@step(enable_cache=False)
def load_positron_angle_inference_inputs(
    parquet_paths: list[str],
    group_probs_parquet_paths: list[str] | None = None,
    group_splitter_parquet_paths: list[str] | None = None,
    endpoint_parquet_paths: list[str] | None = None,
    event_splitter_parquet_paths: list[str] | None = None,
    pion_stop_parquet_paths: list[str] | None = None,
    pipeline_config: dict | None = None,
) -> dict:
    service = PositronAngleInferenceInputsService(pipeline_config=pipeline_config)
    return service.execute(
        parquet_paths=parquet_paths,
        group_probs_parquet_paths=group_probs_parquet_paths,
        group_splitter_parquet_paths=group_splitter_parquet_paths,
        endpoint_parquet_paths=endpoint_parquet_paths,
        event_splitter_parquet_paths=event_splitter_parquet_paths,
        pion_stop_parquet_paths=pion_stop_parquet_paths,
    )
