from zenml import step

from pioneerml.common.loader import TrainingBatchBundle
from pioneerml.pipelines.training.event_splitting.services import EventSplitterExportService


@step
def export_event_splitter(
    module,
    dataset: TrainingBatchBundle,
    pipeline_config: dict | None = None,
    hpo_params: dict | None = None,
    metrics: dict | None = None,
) -> dict:
    service = EventSplitterExportService(
        dataset=dataset,
        module=module,
        pipeline_config=pipeline_config,
        hpo_params=hpo_params,
        metrics=metrics,
    )
    return service.execute()
