from zenml import step

from pioneerml.common.pipeline.steps import BaseWriterStep


class GroupSplittingWriterStep(BaseWriterStep):
    step_key = "writer"
    writer_name = "group_splitting"

    def execute(
        self,
        *,
        inputs: dict,
        output_dir: str | None = None,
        output_path: str | None = None,
    ) -> dict:
        _ = inputs
        return self.build_writer_setup(output_dir=output_dir, output_path=output_path)


@step(name="load_group_splitter_writer", enable_cache=False)
def load_group_splitter_writer_step(
    inputs: dict,
    output_dir: str | None = None,
    output_path: str | None = None,
    pipeline_config: dict | None = None,
) -> dict:
    return GroupSplittingWriterStep(pipeline_config=pipeline_config).execute(
        inputs=inputs,
        output_dir=output_dir,
        output_path=output_path,
    )

