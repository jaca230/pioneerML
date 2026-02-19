from zenml import step

from pioneerml.pipelines.training.event_splitting.dataset import EventSplitterDataset
from pioneerml.pipelines.training.event_splitting.services import EventSplitterTrainingService


@step
def train_event_splitter(
    dataset: EventSplitterDataset,
    pipeline_config: dict | None = None,
    hpo_params: dict | None = None,
):
    service = EventSplitterTrainingService(
        dataset=dataset,
        pipeline_config=pipeline_config,
        hpo_params=hpo_params,
    )
    return service.execute()
