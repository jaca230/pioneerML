from zenml import step

from pioneerml.common.pipeline_utils.export import TorchscriptExporter
from pioneerml.pipelines.training.endpoint_regression.dataset import EndpointRegressorDataset
from pioneerml.pipelines.training.endpoint_regression.steps.config import resolve_step_config


_EXPORTER = TorchscriptExporter()


def _build_export_example(dataset: EndpointRegressorDataset):
    data = dataset.data
    if hasattr(data, "batch"):
        return (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
            data.u,
            data.group_probs,
            data.splitter_probs,
        )
    return None


@step
def export_endpoint_regressor(
    module,
    dataset: EndpointRegressorDataset,
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
        default_export_dir="trained_models/endpoint_regressor",
        default_prefix="endpoint_regressor",
        example_builder=_build_export_example,
    )
