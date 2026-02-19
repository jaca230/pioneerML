from zenml import step

from pioneerml.pipelines.inference.event_splitting.services import EventSplitterInferenceModelLoaderService


@step(enable_cache=False)
def load_event_splitter_model(
    model_path: str | None = None,
    pipeline_config: dict | None = None,
) -> dict:
    service = EventSplitterInferenceModelLoaderService(pipeline_config=pipeline_config)
    return service.execute(model_path=model_path)
