from zenml import step

from pioneerml.pipelines.training.event_splitting.dataset import EventSplitterDataset
from pioneerml.pipelines.training.event_splitting.services import EventSplitterHPOService


@step
def tune_event_splitter(
    dataset: EventSplitterDataset,
    pipeline_config: dict | None = None,
) -> dict:
    service = EventSplitterHPOService(
        dataset=dataset,
        pipeline_config=pipeline_config,
    )
    return service.execute()
