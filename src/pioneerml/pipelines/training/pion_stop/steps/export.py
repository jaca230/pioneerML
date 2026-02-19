from zenml import step

from pioneerml.pipelines.training.pion_stop.dataset import PionStopDataset
from pioneerml.pipelines.training.pion_stop.services import PionStopExportService


@step
def export_pion_stop(
    module,
    dataset: PionStopDataset,
    pipeline_config: dict | None = None,
    hpo_params: dict | None = None,
    metrics: dict | None = None,
) -> dict:
    service = PionStopExportService(
        dataset=dataset,
        module=module,
        pipeline_config=pipeline_config,
        hpo_params=hpo_params,
        metrics=metrics,
    )
    return service.execute()
