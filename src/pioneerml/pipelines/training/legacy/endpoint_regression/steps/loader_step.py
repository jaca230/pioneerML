from zenml import step

from pioneerml.common.pipeline.steps import BaseLoaderFactoryInitStep


class EndpointRegressorLoaderStep(BaseLoaderFactoryInitStep):
    step_key = "loader"

    def default_config(self) -> dict:
        return {
            "loader": {
                "type": "endpoint_regression",
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


@step(name="load_endpoint_regressor_dataset", enable_cache=False)
def load_endpoint_regressor_dataset_step(
    pipeline_config: dict | None = None,
):
    return EndpointRegressorLoaderStep(pipeline_config=pipeline_config).execute()
