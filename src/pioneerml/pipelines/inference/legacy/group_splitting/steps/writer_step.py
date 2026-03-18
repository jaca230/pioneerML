from zenml import step

from pioneerml.common.pipeline.steps import BaseWriterFactoryInitStep


class GroupSplittingWriterStep(BaseWriterFactoryInitStep):
    step_key = "writer_factory_init"

    def default_config(self) -> dict:
        return {"writer_name": "group_splitting"}


@step(name="load_group_splitter_writer", enable_cache=False)
def load_group_splitter_writer_step(
    output_dir: str | None = None,
    output_path: str | None = None,
    pipeline_config: dict | None = None,
) -> dict:
    return GroupSplittingWriterStep(pipeline_config=pipeline_config).execute(
        output_dir=output_dir,
        output_path=output_path,
    )
