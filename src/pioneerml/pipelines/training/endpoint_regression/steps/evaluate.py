from zenml import step

from pioneerml.common.loader import TrainingBatchBundle
from pioneerml.pipelines.training.endpoint_regression.services import EndpointRegressorEvaluationService


@step
def evaluate_endpoint_regressor(
    module,
    dataset: TrainingBatchBundle,
    pipeline_config: dict | None = None,
) -> dict:
    service = EndpointRegressorEvaluationService(
        dataset=dataset,
        module=module,
        pipeline_config=pipeline_config,
    )
    return service.execute()
