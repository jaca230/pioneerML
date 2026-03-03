from pathlib import Path

from zenml import step

from pioneerml.common.pipeline.steps import BaseModelLoaderStep


class EndpointRegressorInferenceModelLoaderStep(BaseModelLoaderStep):
    step_key = "model_loader"

    @staticmethod
    def candidate_model_dirs() -> list[Path]:
        this_file = Path(__file__).resolve()
        return BaseModelLoaderStep.candidate_model_dirs(
            model_subdir="endpoint_regressor",
            this_file=this_file,
            repo_parents_up=8,
        )

    def execute(
        self,
        *,
        model_path: str | None = None,
    ) -> dict:
        cfg = self.get_config()
        selected = model_path or cfg.get("model_path")
        resolved = self.resolve_model_path(
            selected_path=selected,
            candidate_dirs=self.candidate_model_dirs(),
            glob_pattern="*_torchscript.pt",
        )
        return {"model_path": resolved}


@step(name="load_endpoint_regressor_model", enable_cache=False)
def load_endpoint_regressor_model_step(
    model_path: str | None = None,
    pipeline_config: dict | None = None,
) -> dict:
    return EndpointRegressorInferenceModelLoaderStep(pipeline_config=pipeline_config).execute(model_path=model_path)
