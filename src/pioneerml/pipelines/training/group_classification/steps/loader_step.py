
from zenml import step

from pioneerml.common.loader import GroupClassifierGraphLoaderFactory, TrainingBatchBundle
from pioneerml.common.pipeline.steps import BaseLoaderStep
from pioneerml.common.zenml.materializers import TrainingBatchBundleMaterializer


class GroupClassifierLoaderStep(BaseLoaderStep):
    step_key = "loader"

    def default_config(self) -> dict:
        return {"config_json": {}}

    def execute(
        self,
        *,
        parquet_paths: list[str],
    ) -> TrainingBatchBundle:
        cfg = self.get_config()
        config_json = dict(cfg.get("config_json") or {})

        loader_factory = GroupClassifierGraphLoaderFactory(parquet_paths=[str(p) for p in parquet_paths])
        loader = loader_factory.build_loader(loader_params=dict(config_json))
        inputs, targets = loader.empty_data()
        inputs.source_parquet_paths = list(loader.parquet_paths)
        return TrainingBatchBundle(
            inputs=inputs,
            targets=targets,
            loader_factory=loader_factory,
            loader=loader_factory,
        )


@step(
    name="load_group_classifier_dataset",
    enable_cache=False,
    output_materializers=TrainingBatchBundleMaterializer,
)
def load_group_classifier_dataset_step(
    parquet_paths: list[str],
    pipeline_config: dict | None = None,
) -> TrainingBatchBundle:
    return GroupClassifierLoaderStep(pipeline_config=pipeline_config).execute(parquet_paths=parquet_paths)
