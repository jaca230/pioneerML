from zenml import step

from pioneerml.common.loader import TrainingBatchBundle
from pioneerml.pipelines.training.endpoint_regression.services import EndpointRegressorExportService


@step
def export_endpoint_regressor(
    module,
    dataset: TrainingBatchBundle,
    pipeline_config: dict | None = None,
    hpo_params: dict | None = None,
    metrics: dict | None = None,
) -> dict:
    service = EndpointRegressorExportService(
        dataset=dataset,
        module=module,
        pipeline_config=pipeline_config,
        hpo_params=hpo_params,
        metrics=metrics,
    )
    return service.execute()
