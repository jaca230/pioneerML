from zenml import step

from pioneerml.pipelines.training.pion_stop.dataset import PionStopDataset
from pioneerml.pipelines.training.pion_stop.services import PionStopTrainingService


@step
def train_pion_stop(
    dataset: PionStopDataset,
    pipeline_config: dict | None = None,
    hpo_params: dict | None = None,
):
    service = PionStopTrainingService(
        dataset=dataset,
        pipeline_config=pipeline_config,
        hpo_params=hpo_params,
    )
    return service.execute()
