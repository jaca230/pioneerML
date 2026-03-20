from zenml import step

from pioneerml.common.pipeline.steps import BaseLoaderFactoryInitStep


class GroupClassifierLoaderStep(BaseLoaderFactoryInitStep):
    step_key = "loader"

    def default_config(self) -> dict:
        return {
            "loader": {
                "type": "group_classifier",
                "config": {
                    "input_sources_spec": {
                        "main_sources": [],
                        "optional_sources_by_name": {},
                        "source_type": "file",
                    },
                    "input_backend_name": "parquet",
                    "mode": "train",
                },
            }
        }


@step(name="load_group_classifier_dataset", enable_cache=False)
def load_group_classifier_dataset_step(
    pipeline_config: dict | None = None,
):
    return GroupClassifierLoaderStep(pipeline_config=pipeline_config).execute()
