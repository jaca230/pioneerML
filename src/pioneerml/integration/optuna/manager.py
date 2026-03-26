from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Tuple

import optuna


class OptunaStudyManager:
    """
    Object-oriented helper to resolve Optuna storage and create/load studies.
    """

    def __init__(
        self,
        *,
        project_root: Path | str | None = None,
        study_name: str = "optuna_study",
        direction: str = "maximize",
        storage: str | None = None,
        fallback_dir: str | None = None,
        allow_schema_fallback: bool = True,
    ):
        self.project_root = self._resolve_project_root(project_root)
        self.study_name = study_name
        self.direction = direction
        self.explicit_storage = storage
        if fallback_dir:
            fb = Path(fallback_dir).expanduser()
            if not fb.is_absolute():
                fb = self.project_root / fb
            self.fallback_dir = fb.resolve()
        else:
            self.fallback_dir = (self.project_root / ".optuna").resolve()
        self.allow_schema_fallback = allow_schema_fallback

    @staticmethod
    def _resolve_project_root(project_root: Path | str | None) -> Path:
        if project_root is not None:
            return Path(project_root).expanduser().resolve()
        start = Path.cwd().resolve()
        sentinels = ("pyproject.toml", ".zen")
        for parent in (start, *start.parents):
            if any((parent / s).exists() for s in sentinels):
                return parent
        return start

    def _normalize_storage_url(self, storage: str) -> str:
        raw = str(storage).strip()
        if raw == "":
            return raw
        if not raw.startswith("sqlite:///"):
            return raw

        path_part = raw[len("sqlite:///") :]
        # Keep special in-memory sqlite URI untouched.
        if path_part == ":memory:":
            return raw

        db_path = Path(path_part).expanduser()
        if not db_path.is_absolute():
            db_path = (self.project_root / db_path).resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{db_path.as_posix()}"

    def _zenml_store_url(self) -> str | None:
        try:
            from zenml.client import Client  # type: ignore
        except Exception:
            return None
        try:
            return Client().zen_store.config.url  # type: ignore[attr-defined]
        except Exception:
            return None

    def resolve_storage(self) -> str:
        """
        Resolve an Optuna storage URI by checking:
        1) explicit storage provided to the manager
        2) env vars (OPTUNA_STORAGE, ZENML_OPTUNA_STORAGE)
        3) local sqlite fallback under .optuna/<study>.db
        """
        if self.explicit_storage:
            return self._normalize_storage_url(self.explicit_storage)

        env_storage = os.environ.get("OPTUNA_STORAGE") or os.environ.get("ZENML_OPTUNA_STORAGE")
        if env_storage:
            return self._normalize_storage_url(env_storage)

        self.fallback_dir.mkdir(parents=True, exist_ok=True)
        return self._normalize_storage_url(f"sqlite:///{self.fallback_dir}/{self.study_name}.db")

    def create_or_load(self, **study_kwargs: Any) -> Tuple[optuna.study.Study, str]:
        """
        Create or load an Optuna study using resolved storage. Returns (study, storage_used).
        """
        storage_used = self.resolve_storage()
        try:
            study = optuna.create_study(
                study_name=self.study_name,
                storage=storage_used,
                direction=self.direction,
                load_if_exists=True,
                **study_kwargs,
            )
            return study, storage_used
        except Exception as e:
            if self.allow_schema_fallback:
                # fallback to in-memory study if storage is incompatible
                study = optuna.create_study(
                    study_name=self.study_name,
                    storage=None,
                    direction=self.direction,
                    load_if_exists=False,
                    **study_kwargs,
                )
                return study, "in-memory"
            raise
