from zenml import step

from pioneerml.common.loader import TrainingBatchBundle
from pioneerml.pipelines.training.endpoint_regression.services import EndpointRegressorHPOService


@step
def tune_endpoint_regressor(
    dataset: TrainingBatchBundle,
    pipeline_config: dict | None = None,
) -> dict:
    service = EndpointRegressorHPOService(
        dataset=dataset,
        pipeline_config=pipeline_config,
    )
    return service.execute()
