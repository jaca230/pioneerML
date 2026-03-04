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
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self.study_name = study_name
        self.direction = direction
        self.explicit_storage = storage
        self.fallback_dir = Path(fallback_dir) if fallback_dir else self.project_root / ".optuna"
        self.allow_schema_fallback = allow_schema_fallback

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
            return self.explicit_storage

        env_storage = os.environ.get("OPTUNA_STORAGE") or os.environ.get("ZENML_OPTUNA_STORAGE")
        if env_storage:
            return env_storage

        self.fallback_dir.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{self.fallback_dir}/{self.study_name}.db"

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
        except RuntimeError as e:
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
