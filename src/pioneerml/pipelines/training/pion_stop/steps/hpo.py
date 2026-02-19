from zenml import step

from pioneerml.pipelines.training.pion_stop.dataset import PionStopDataset
from pioneerml.pipelines.training.pion_stop.services import PionStopHPOService


@step
def tune_pion_stop(
    dataset: PionStopDataset,
    pipeline_config: dict | None = None,
) -> dict:
    service = PionStopHPOService(
        dataset=dataset,
        pipeline_config=pipeline_config,
    )
    return service.execute()
