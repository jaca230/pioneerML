from zenml import step

from pioneerml.pipelines.inference.event_splitting.services import EventSplitterSavePredictionsService


@step(enable_cache=False)
def save_event_splitter_predictions(
    inference_outputs: dict,
    output_dir: str | None = None,
    output_path: str | None = None,
    metrics_path: str | None = None,
    pipeline_config: dict | None = None,
) -> dict:
    service = EventSplitterSavePredictionsService(pipeline_config=pipeline_config)
    return service.execute(
        inference_outputs=inference_outputs,
        output_dir=output_dir,
        output_path=output_path,
        metrics_path=metrics_path,
    )
