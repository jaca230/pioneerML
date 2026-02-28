from zenml import step

from pioneerml.common.loader import TrainingBatchBundle
from pioneerml.pipelines.training.pion_stop.services import PionStopEvaluationService


@step
def evaluate_pion_stop(
    module,
    dataset: TrainingBatchBundle,
    pipeline_config: dict | None = None,
) -> dict:
    service = PionStopEvaluationService(
        dataset=dataset,
        module=module,
        pipeline_config=pipeline_config,
    )
    return service.execute()
