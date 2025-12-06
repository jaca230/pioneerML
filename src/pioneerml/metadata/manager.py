from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch

try:
    # Reuse project root helper if available
    from pioneerml.zenml.utils import find_project_root  # type: ignore
except Exception:  # pragma: no cover
    def find_project_root(start: Path | None = None) -> Path:
        return Path(start or Path.cwd()).resolve()


def timestamp_now() -> str:
    """Return a filesystem-friendly UTC timestamp."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


@dataclass
class TrainingMetadata:
    model_type: str
    timestamp: str
    run_name: str | None = None
    dataset_info: Dict[str, Any] | None = None
    training_config: Dict[str, Any] | None = None
    model_architecture: Dict[str, Any] | None = None
    best_hyperparameters: Dict[str, Any] | None = None
    best_score: float | None = None
    n_trials: int | None = None
    epochs_run: int | None = None
    hyperparameter_history: Optional[list[dict[str, Any]]] = None  # Deprecated: use optuna_storage + optuna_study_name instead
    optuna_storage: str | None = None  # Optuna storage URI (e.g., "sqlite:///path/to/db")
    optuna_study_name: str | None = None  # Optuna study name within the storage
    artifact_paths: Dict[str, str] | None = None
    extra: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TrainingMetadata":
        return TrainingMetadata(**data)


def build_artifact_paths(
    model_type: str,
    *,
    timestamp: str | None = None,
    run_name: str | None = None,
    root: Path | str | None = None,
) -> dict[str, Path]:
    """
    Build standard artifact paths under trained_models/<model_type>/.
    """
    ts = timestamp or timestamp_now()
    root_path = Path(root or find_project_root()) / "trained_models" / model_type.lower()
    root_path.mkdir(parents=True, exist_ok=True)

    prefix = f"{model_type.lower()}_{ts}"
    if run_name:
        # make run name filesystem-friendly
        safe_run = "".join(c if c.isalnum() or c in "-_." else "_" for c in run_name)
        prefix = f"{prefix}_{safe_run}"

    return {
        "dir": root_path,
        "state_dict": root_path / f"{prefix}_state_dict.pt",
        "metadata": root_path / f"{prefix}_metadata.json",
        "full_checkpoint": root_path / f"{prefix}_checkpoint.pt",
    }


def save_model_and_metadata(
    model: torch.nn.Module | None,
    metadata: TrainingMetadata,
    *,
    state_dict_only: bool = True,
    root: Path | str | None = None,
) -> dict[str, Path]:
    """
    Save a model (state_dict or full checkpoint) plus standardized metadata JSON.
    Returns the artifact paths used.
    """
    paths = build_artifact_paths(
        metadata.model_type, timestamp=metadata.timestamp, run_name=metadata.run_name, root=root
    )

    artifact_paths: dict[str, str] = {}
    if model is not None:
        if state_dict_only:
            torch.save(model.state_dict(), paths["state_dict"])
            artifact_paths["state_dict"] = str(paths["state_dict"])
        else:
            torch.save(model, paths["full_checkpoint"])
            artifact_paths["full_checkpoint"] = str(paths["full_checkpoint"])

    # Merge in resolved artifact paths to metadata for easy reload later
    merged_meta = metadata.to_dict()
    existing = merged_meta.get("artifact_paths") or {}
    existing.update(artifact_paths)
    merged_meta["artifact_paths"] = existing

    with open(paths["metadata"], "w", encoding="utf-8") as f:
        json.dump(merged_meta, f, indent=2)

    return paths


def load_metadata(path: Path | str) -> TrainingMetadata:
    """Load TrainingMetadata from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return TrainingMetadata.from_dict(data)


# --- Optuna helpers ---------------------------------------------------------
def serialize_optuna_study(
    study: Any,
    *,
    include_failed: bool = False,
    include_intermediate_values: bool = False,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """
    Convert an Optuna study into a lightweight list of trial dicts suitable
    for storing in TrainingMetadata.hyperparameter_history.
    """
    try:
        import optuna  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Optuna is required to serialize study history.") from e

    trials = study.get_trials(deepcopy=False)
    # sort by trial number for reproducibility
    trials = sorted(trials, key=lambda t: t.number)
    if not include_failed:
        trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    if limit is not None:
        trials = trials[-limit:]

    history: list[dict[str, Any]] = []
    for t in trials:
        entry = {
            "number": t.number,
            "state": t.state.name,
            "value": t.value,
            "params": dict(t.params),
            "user_attrs": dict(t.user_attrs),
            "datetime_start": t.datetime_start.isoformat() if t.datetime_start else None,
            "datetime_complete": t.datetime_complete.isoformat() if t.datetime_complete else None,
        }
        if include_intermediate_values:
            entry["intermediate_values"] = dict(t.intermediate_values)
        history.append(entry)
    return history
