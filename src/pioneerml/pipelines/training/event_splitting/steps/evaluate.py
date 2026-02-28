from zenml import step

from pioneerml.common.loader import TrainingBatchBundle
from pioneerml.pipelines.training.event_splitting.services import EventSplitterEvaluationService


@step
def evaluate_event_splitter(
    module,
    dataset: TrainingBatchBundle,
    pipeline_config: dict | None = None,
) -> dict:
    service = EventSplitterEvaluationService(
        dataset=dataset,
        module=module,
        pipeline_config=pipeline_config,
    )
    return service.execute()
