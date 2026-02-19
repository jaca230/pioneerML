from zenml import step

from pioneerml.pipelines.training.event_splitting.dataset import EventSplitterDataset
from pioneerml.pipelines.training.event_splitting.services import EventSplitterEvaluationService


@step
def evaluate_event_splitter(
    module,
    dataset: EventSplitterDataset,
    pipeline_config: dict | None = None,
) -> dict:
    service = EventSplitterEvaluationService(
        dataset=dataset,
        module=module,
        pipeline_config=pipeline_config,
    )
    return service.execute()
