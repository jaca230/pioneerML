from __future__ import annotations

from pathlib import Path

from .payloads import ModelLoaderStepPayload
from .resolvers.path_resolver import ModelPathResolver

from ..base_pipeline_step import BasePipelineStep


class BaseModelLoaderStep(BasePipelineStep):
    resolver_classes = (ModelPathResolver,)
    DEFAULT_CONFIG = {"model_path": None}

    @staticmethod
    def candidate_model_dirs(*, model_subdir: str, this_file: Path, repo_parents_up: int) -> list[Path]:
        return ModelPathResolver.candidate_model_dirs(
            model_subdir=model_subdir,
            this_file=this_file,
            repo_parents_up=repo_parents_up,
        )

    @staticmethod
    def resolve_model_path(
        *,
        selected_path: str | None,
        candidate_dirs: list[Path],
        glob_pattern: str = "*_torchscript.pt",
    ) -> str:
        return ModelPathResolver.resolve_model_path(
            selected_path=selected_path,
            candidate_dirs=candidate_dirs,
            glob_pattern=glob_pattern,
        )

    def candidate_model_dirs(self) -> list[Path]:
        raise NotImplementedError(f"{self.__class__.__name__} must implement candidate_model_dirs().")

    def resolve_model_glob_pattern(self) -> str:
        return "*_torchscript.pt"

    def build_payload(self, *, model_path: str) -> ModelLoaderStepPayload:
        return ModelLoaderStepPayload(model_path=str(model_path))

    def _execute(self, *, model_path: str | None = None) -> ModelLoaderStepPayload:
        selected = model_path or self.config_json.get("model_path")
        resolved = self.resolve_model_path(
            selected_path=(None if selected is None else str(selected)),
            candidate_dirs=self.candidate_model_dirs(),
            glob_pattern=self.resolve_model_glob_pattern(),
        )
        return self.build_payload(model_path=resolved)
