from zenml import step

from pioneerml.pipelines.training.positron_angle.dataset import PositronAngleDataset
from pioneerml.pipelines.training.positron_angle.services import PositronAngleExportService


@step
def export_positron_angle(
    module,
    dataset: PositronAngleDataset,
    pipeline_config: dict | None = None,
    hpo_params: dict | None = None,
    metrics: dict | None = None,
) -> dict:
    service = PositronAngleExportService(
        dataset=dataset,
        module=module,
        pipeline_config=pipeline_config,
        hpo_params=hpo_params,
        metrics=metrics,
    )
    return service.execute()
