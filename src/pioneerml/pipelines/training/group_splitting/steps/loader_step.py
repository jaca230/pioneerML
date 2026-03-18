from zenml import step

from pioneerml.common.pipeline.steps import BaseLoaderFactoryInitStep


class GroupSplitterLoaderStep(BaseLoaderFactoryInitStep):
    step_key = "loader"

    def default_config(self) -> dict:
        return {"loader_name": "group_splitter"}


@step(name="load_group_splitter_dataset", enable_cache=False)
def load_group_splitter_dataset_step(
    pipeline_config: dict | None = None,
):
    return GroupSplitterLoaderStep(pipeline_config=pipeline_config).execute()
