
from zenml import step

from pioneerml.common.loader import GroupSplitterGraphLoaderFactory, BatchBundle
from pioneerml.common.pipeline.steps import BaseLoaderStep
from pioneerml.common.zenml.materializers import BatchBundleMaterializer


class GroupSplitterLoaderStep(BaseLoaderStep):
    step_key = "loader"

    def default_config(self) -> dict:
        return {"config_json": {}}

    def execute(
        self,
        *,
        parquet_input_set: dict,
    ) -> BatchBundle:
        cfg = self.get_config()
        config_json = dict(cfg.get("config_json") or {})
        parquet_inputs = self.resolve_parquet_input_set(parquet_input_set)
        loader_factory = GroupSplitterGraphLoaderFactory(
            parquet_inputs=parquet_inputs,
        )
        loader = loader_factory.build_loader(loader_params=dict(config_json))
        data = loader.empty_data()
        data.source_parquet_paths = list(loader.parquet_paths)
        return BatchBundle(
            data=data,
            loader_factory=loader_factory,
            loader=loader_factory,
        )


@step(
    name="load_group_splitter_dataset",
    enable_cache=False,
    output_materializers=BatchBundleMaterializer,
)
def load_group_splitter_dataset_step(
    parquet_input_set: dict,
    pipeline_config: dict | None = None,
) -> BatchBundle:
    return GroupSplitterLoaderStep(pipeline_config=pipeline_config).execute(
        parquet_input_set=parquet_input_set,
    )
