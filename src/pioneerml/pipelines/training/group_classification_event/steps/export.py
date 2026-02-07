from zenml import step

from pioneerml.common.pipeline_utils.export import TorchscriptExporter
from pioneerml.pipelines.training.group_classification_event.dataset import GroupClassifierEventDataset
from pioneerml.pipelines.training.group_classification_event.steps.config import resolve_step_config


_EXPORTER = TorchscriptExporter()


def _build_export_example(dataset: GroupClassifierEventDataset):
    data = dataset.data
    if hasattr(data, "batch"):
        return (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
            data.group_ptr,
            data.time_group_ids,
        )
    return None


@step
def export_group_classifier_event(
    module,
    dataset: GroupClassifierEventDataset,
    pipeline_config: dict | None = None,
    hpo_params: dict | None = None,
    metrics: dict | None = None,
) -> dict:
    cfg = resolve_step_config(pipeline_config, "export") or {}
    return _EXPORTER.export(
        module=module,
        dataset=dataset,
        cfg=cfg,
        hpo_params=hpo_params,
        metrics=metrics,
        default_export_dir="trained_models/groupclassifier_event",
        default_prefix="groupclassifier_event",
        example_builder=_build_export_example,
    )
