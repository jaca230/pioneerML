from zenml import step

from pioneerml.common.loader import GroupSplitterGraphLoaderFactory
from pioneerml.common.zenml.materializers import GroupSplitterDatasetMaterializer
from pioneerml.pipelines.training.group_splitting.dataset import GroupSplitterDataset


@step(enable_cache=False, output_materializers=GroupSplitterDatasetMaterializer)
def load_group_splitter_dataset(
    parquet_paths: list[str],
    group_probs_parquet_paths: list[str] | None = None,
    pipeline_config: dict | None = None,
) -> GroupSplitterDataset:
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

    loader_factory = GroupSplitterGraphLoaderFactory(
        parquet_paths=[str(p) for p in parquet_paths],
        group_probs_parquet_paths=[str(p) for p in group_probs_parquet_paths] if group_probs_parquet_paths else None,
    )
    loader = loader_factory.build_loader(loader_params=dict(config_json))
    data, targets = loader.empty_data()
    data.source_parquet_paths = list(loader.parquet_paths)
    if loader.group_probs_parquet_paths is not None:
        data.group_probs_parquet_paths = list(loader.group_probs_parquet_paths)
    return GroupSplitterDataset(data=data, targets=targets, loader_factory=loader_factory, loader=loader_factory)
