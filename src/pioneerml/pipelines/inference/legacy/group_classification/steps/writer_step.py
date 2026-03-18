from zenml import step

from pioneerml.common.pipeline.steps import BaseWriterFactoryInitStep


class GroupClassificationWriterStep(BaseWriterFactoryInitStep):
    step_key = "writer_factory_init"

    def default_config(self) -> dict:
        return {"writer_name": "group_classification"}


@step(name="load_group_classifier_writer", enable_cache=False)
def load_group_classifier_writer_step(
    output_dir: str | None = None,
    output_path: str | None = None,
    pipeline_config: dict | None = None,
) -> dict:
    return GroupClassificationWriterStep(pipeline_config=pipeline_config).execute(
        output_dir=output_dir,
        output_path=output_path,
    )
