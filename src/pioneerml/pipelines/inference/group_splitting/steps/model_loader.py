from zenml import step

from pioneerml.pipelines.inference.group_splitting.services import GroupSplitterInferenceModelLoaderService


@step(enable_cache=False)
def load_group_splitter_model(
    model_path: str | None = None,
    pipeline_config: dict | None = None,
) -> dict:
    service = GroupSplitterInferenceModelLoaderService(pipeline_config=pipeline_config)
    return service.execute(model_path=model_path)
