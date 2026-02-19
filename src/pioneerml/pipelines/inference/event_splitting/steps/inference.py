from zenml import step

from pioneerml.pipelines.inference.event_splitting.services import EventSplitterInferenceRunService


@step(enable_cache=False)
def run_event_splitter_inference(
    model_info: dict,
    inputs: dict,
    pipeline_config: dict | None = None,
) -> dict:
    service = EventSplitterInferenceRunService(pipeline_config=pipeline_config)
    return service.execute(model_info=model_info, inputs=inputs)
