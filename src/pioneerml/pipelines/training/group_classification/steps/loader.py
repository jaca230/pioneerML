from zenml import step

from pioneerml.common.loader import GroupClassifierGraphLoaderFactory
from pioneerml.common.zenml.materializers import GroupClassifierDatasetMaterializer
from pioneerml.pipelines.training.group_classification.dataset import GroupClassifierDataset


@step(enable_cache=False, output_materializers=GroupClassifierDatasetMaterializer)
def load_group_classifier_dataset(
    parquet_paths: list[str],
    pipeline_config: dict | None = None,
) -> GroupClassifierDataset:
    if pipeline_config is not None and not isinstance(pipeline_config, dict):
        raise TypeError(f"Expected dict for pipeline_config, got {type(pipeline_config).__name__}.")
    step_config = {}
    if isinstance(pipeline_config, dict):
        raw = pipeline_config.get("loader")
        if raw is not None:
            if not isinstance(raw, dict):
                raise TypeError(f"Expected dict for 'loader' config, got {type(raw).__name__}.")
            step_config = dict(raw)
    config_json = dict(step_config.get("config_json") or {})

    loader_factory = GroupClassifierGraphLoaderFactory(parquet_paths=[str(p) for p in parquet_paths])
    loader = loader_factory.build_loader(loader_params=dict(config_json))

    data, targets = loader.empty_data()
    data.source_parquet_paths = list(loader.parquet_paths)
    return GroupClassifierDataset(data=data, targets=targets, loader_factory=loader_factory, loader=loader_factory)
