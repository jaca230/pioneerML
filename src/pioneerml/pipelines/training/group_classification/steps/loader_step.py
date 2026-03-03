
from zenml import step

from pioneerml.common.loader import GroupClassifierGraphLoaderFactory, BatchBundle
from pioneerml.common.pipeline.steps import BaseLoaderStep
from pioneerml.common.zenml.materializers import BatchBundleMaterializer


class GroupClassifierLoaderStep(BaseLoaderStep):
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

        loader_factory = GroupClassifierGraphLoaderFactory(
            parquet_inputs=parquet_inputs
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
    name="load_group_classifier_dataset",
    enable_cache=False,
    output_materializers=BatchBundleMaterializer,
)
def load_group_classifier_dataset_step(
    parquet_input_set: dict,
    pipeline_config: dict | None = None,
) -> BatchBundle:
    return GroupClassifierLoaderStep(pipeline_config=pipeline_config).execute(parquet_input_set=parquet_input_set)
