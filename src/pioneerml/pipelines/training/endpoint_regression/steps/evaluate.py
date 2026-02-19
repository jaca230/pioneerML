from zenml import step

from pioneerml.pipelines.training.endpoint_regression.dataset import EndpointRegressorDataset
from pioneerml.pipelines.training.endpoint_regression.services import EndpointRegressorEvaluationService


@step
def evaluate_endpoint_regressor(
    module,
    dataset: EndpointRegressorDataset,
    pipeline_config: dict | None = None,
) -> dict:
    service = EndpointRegressorEvaluationService(
        dataset=dataset,
        module=module,
        pipeline_config=pipeline_config,
    )
    return service.execute()
