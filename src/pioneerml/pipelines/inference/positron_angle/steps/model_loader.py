from zenml import step

from pioneerml.pipelines.inference.positron_angle.services import (
    PositronAngleInferenceModelLoaderService,
)


@step(enable_cache=False)
def load_positron_angle_model(
    model_path: str | None = None,
    pipeline_config: dict | None = None,
) -> dict:
    service = PositronAngleInferenceModelLoaderService(pipeline_config=pipeline_config)
    return service.execute(model_path=model_path)
