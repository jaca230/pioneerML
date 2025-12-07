"""
Metadata type definitions.

This module contains the core dataclasses for model metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class TrainingMetadata:
    """Metadata for a trained model, including architecture, hyperparameters, and training history."""
    
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
        """Convert to dictionary."""
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TrainingMetadata":
        """Create from dictionary."""
        return TrainingMetadata(**data)


