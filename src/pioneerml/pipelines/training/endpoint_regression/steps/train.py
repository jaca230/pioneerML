from zenml import step

from pioneerml.pipelines.training.endpoint_regression.dataset import EndpointRegressorDataset
from pioneerml.pipelines.training.endpoint_regression.services import EndpointRegressorTrainingService


@step
def train_endpoint_regressor(
    dataset: EndpointRegressorDataset,
    pipeline_config: dict | None = None,
    hpo_params: dict | None = None,
):
    service = EndpointRegressorTrainingService(
        dataset=dataset,
        pipeline_config=pipeline_config,
        hpo_params=hpo_params,
    )
    return service.execute()
