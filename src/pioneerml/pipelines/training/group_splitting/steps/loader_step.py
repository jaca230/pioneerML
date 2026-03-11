
from zenml import step

from pioneerml.common.data_loader import LoaderFactory, BatchBundle
from pioneerml.common.pipeline.steps import BaseLoaderFactoryInitStep
from pioneerml.common.integration.zenml.materializers import BatchBundleMaterializer


class GroupSplitterLoaderStep(BaseLoaderFactoryInitStep):
    step_key = "loader"

    def default_config(self) -> dict:
        return {"config_json": {}}

    def run(
        self,
        *,
        input_source_set: dict,
    ) -> BatchBundle:
        cfg = self.config_json
        config_json = dict(cfg.get("config_json") or {})
        input_sources = self.resolve_input_source_set(input_source_set)
        loader_factory = LoaderFactory(
            loader_name="group_splitter",
            input_sources=input_sources,
        )
        loader = loader_factory.build_loader(loader_params=dict(config_json))
        data = loader.empty_data()
        data.source_main_sources = list(loader.input_sources.main_sources)
        return BatchBundle(
            data=data,
            loader_factory=loader_factory,
            loader=loader,
        )


@step(
    name="load_group_splitter_dataset",
    enable_cache=False,
    output_materializers=BatchBundleMaterializer,
)
def load_group_splitter_dataset_step(
    input_source_set: dict,
    pipeline_config: dict | None = None,
) -> BatchBundle:
    return GroupSplitterLoaderStep(pipeline_config=pipeline_config).execute(
        input_source_set=input_source_set,
    )
