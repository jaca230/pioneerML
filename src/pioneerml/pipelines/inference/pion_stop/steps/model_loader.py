from zenml import step

from pioneerml.pipelines.inference.pion_stop.services import (
    PionStopInferenceModelLoaderService,
)


@step(enable_cache=False)
def load_pion_stop_model(
    model_path: str | None = None,
    pipeline_config: dict | None = None,
) -> dict:
    service = PionStopInferenceModelLoaderService(pipeline_config=pipeline_config)
    return service.execute(model_path=model_path)
