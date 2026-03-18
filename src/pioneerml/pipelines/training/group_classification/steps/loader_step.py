from zenml import step

from pioneerml.common.pipeline.steps import BaseLoaderFactoryInitStep

class GroupClassifierLoaderStep(BaseLoaderFactoryInitStep):
    step_key = "loader"

    def default_config(self) -> dict:
        return {"loader_name": "group_classifier"}


@step(name="load_group_classifier_dataset", enable_cache=False)
def load_group_classifier_dataset_step(
    pipeline_config: dict | None = None,
):
    return GroupClassifierLoaderStep(pipeline_config=pipeline_config).execute()
