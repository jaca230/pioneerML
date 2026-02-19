from zenml import step

from pioneerml.pipelines.inference.endpoint_regression.services import (
    EndpointRegressorInferenceModelLoaderService,
)


@step(enable_cache=False)
def load_endpoint_regressor_model(
    model_path: str | None = None,
    pipeline_config: dict | None = None,
) -> dict:
    service = EndpointRegressorInferenceModelLoaderService(pipeline_config=pipeline_config)
    return service.execute(model_path=model_path)
