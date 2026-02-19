from __future__ import annotations

from pathlib import Path

from pioneerml.common.pipeline.services import BaseModelLoaderService


class GroupSplitterInferenceModelLoaderService(BaseModelLoaderService):
    step_key = "model_loader"

    @staticmethod
    def candidate_model_dirs() -> list[Path]:
        this_file = Path(__file__).resolve()
        return BaseModelLoaderService.candidate_model_dirs(
            model_subdir="groupsplitter",
            this_file=this_file,
            repo_parents_up=8,
        )

    def execute(self, *, model_path: str | None) -> dict:
        cfg = self.get_config()
        selected = model_path or cfg.get("model_path")
        resolved = self.resolve_model_path(
            selected_path=selected,
            candidate_dirs=self.candidate_model_dirs(),
            glob_pattern="*_torchscript.pt",
        )
        return {"model_path": resolved}
