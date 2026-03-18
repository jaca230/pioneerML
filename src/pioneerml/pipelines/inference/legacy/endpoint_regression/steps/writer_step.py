from zenml import step

from pioneerml.common.pipeline.steps import BaseWriterFactoryInitStep


class EndpointRegressionWriterStep(BaseWriterFactoryInitStep):
    step_key = "writer_factory_init"

    def default_config(self) -> dict:
        return {"writer_name": "endpoint_regression"}


@step(name="load_endpoint_regressor_writer", enable_cache=False)
def load_endpoint_regressor_writer_step(
    output_dir: str | None = None,
    output_path: str | None = None,
    pipeline_config: dict | None = None,
) -> dict:
    return EndpointRegressionWriterStep(pipeline_config=pipeline_config).execute(
        output_dir=output_dir,
        output_path=output_path,
    )
