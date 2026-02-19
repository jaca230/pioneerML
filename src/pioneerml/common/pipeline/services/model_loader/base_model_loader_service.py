from __future__ import annotations

from pathlib import Path

from ..base_pipeline_service import BasePipelineService


class BaseModelLoaderService(BasePipelineService):
    def default_config(self) -> dict:
        return {}

    @staticmethod
    def candidate_model_dirs(*, model_subdir: str, this_file: Path, repo_parents_up: int) -> list[Path]:
        repo_root = this_file.parents[repo_parents_up] if len(this_file.parents) > repo_parents_up else this_file.parent
        cwd = Path.cwd().resolve()
        candidates = [
            cwd / "trained_models" / model_subdir,
            repo_root / "trained_models" / model_subdir,
            Path(f"/workspace/trained_models/{model_subdir}"),
        ]
        uniq: list[Path] = []
        seen: set[str] = set()
        for p in candidates:
            key = str(p)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(p)
        return uniq

    @staticmethod
    def resolve_model_path(
        *,
        selected_path: str | None,
        candidate_dirs: list[Path],
        glob_pattern: str = "*_torchscript.pt",
    ) -> str:
        if selected_path is not None:
            resolved = Path(selected_path).expanduser().resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Model not found: {resolved}")
            return str(resolved)

        candidates: list[Path] = []
        searched: list[str] = []
        for model_dir in candidate_dirs:
            searched.append(str(model_dir))
            if not model_dir.exists():
                continue
            candidates.extend(model_dir.glob(glob_pattern))
        candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError("No torchscript models found. Searched directories: " + ", ".join(searched))
        return str(candidates[0].resolve())
