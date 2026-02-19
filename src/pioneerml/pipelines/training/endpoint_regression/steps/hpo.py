from zenml import step

from pioneerml.pipelines.training.endpoint_regression.dataset import EndpointRegressorDataset
from pioneerml.pipelines.training.endpoint_regression.services import EndpointRegressorHPOService


@step
def tune_endpoint_regressor(
    dataset: EndpointRegressorDataset,
    pipeline_config: dict | None = None,
) -> dict:
    service = EndpointRegressorHPOService(
        dataset=dataset,
        pipeline_config=pipeline_config,
    )
    return service.execute()
