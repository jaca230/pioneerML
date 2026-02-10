from zenml import step

from pioneerml.common.pipeline_utils.export import TorchscriptExporter
from pioneerml.pipelines.training.group_classification.dataset import GroupClassifierDataset
from pioneerml.pipelines.training.group_classification.steps.config import resolve_step_config


_EXPORTER = TorchscriptExporter()


def _build_export_example(dataset: GroupClassifierDataset):
    data = dataset.data
    if hasattr(data, "batch") and hasattr(data, "u"):
        return (data.x, data.edge_index, data.edge_attr, data.batch, data.u)
    return None


@step
def export_group_classifier(
    module,
    dataset: GroupClassifierDataset,
    pipeline_config: dict | None = None,
    hpo_params: dict | None = None,
    metrics: dict | None = None,
) -> dict:
    cfg = resolve_step_config(pipeline_config, "export") or {}
    return _EXPORTER.export(
        module=module,
        dataset=dataset,
        cfg=cfg,
        pipeline_config=pipeline_config,
        hpo_params=hpo_params,
        metrics=metrics,
        default_export_dir="trained_models/groupclassifier",
        default_prefix="groupclassifier",
        example_builder=_build_export_example,
    )
