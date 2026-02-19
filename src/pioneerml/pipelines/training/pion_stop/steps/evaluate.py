from zenml import step

from pioneerml.pipelines.training.pion_stop.dataset import PionStopDataset
from pioneerml.pipelines.training.pion_stop.services import PionStopEvaluationService


@step
def evaluate_pion_stop(
    module,
    dataset: PionStopDataset,
    pipeline_config: dict | None = None,
) -> dict:
    service = PionStopEvaluationService(
        dataset=dataset,
        module=module,
        pipeline_config=pipeline_config,
    )
    return service.execute()
