from zenml import step

from pioneerml.common.pipeline.steps import BaseWriterStep


class GroupClassificationWriterStep(BaseWriterStep):
    step_key = "writer"
    writer_name = "group_classification"

    def run(
        self,
        *,
        inputs: dict,
        output_dir: str | None = None,
        output_path: str | None = None,
    ) -> dict:
        _ = inputs
        return self.build_writer_setup(output_dir=output_dir, output_path=output_path)


@step(name="load_group_classifier_writer", enable_cache=False)
def load_group_classifier_writer_step(
    inputs: dict,
    output_dir: str | None = None,
    output_path: str | None = None,
    pipeline_config: dict | None = None,
) -> dict:
    return GroupClassificationWriterStep(pipeline_config=pipeline_config).execute(
        inputs=inputs,
        output_dir=output_dir,
        output_path=output_path,
    )

