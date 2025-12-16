"""
Utilities for saving/loading model artifacts and standardized training metadata.

This package uses an object-oriented design:
- MetadataManager: Main class for managing checkpoints, artifacts, and models
  - checkpoints: CheckpointManager instance for checkpoint operations
  - artifacts: ArtifactManager instance for artifact operations
- types: Core dataclasses (TrainingMetadata)
- utils: Utility functions (timestamp_now, optuna serialization)
- loaders: Model loaders for reconstructing models from checkpoints

Usage:
    from pioneerml.metadata import MetadataManager
    
    manager = MetadataManager()
    checkpoints = manager.print_checkpoints("GroupClassifier")
    model, metadata = manager.load_model("GroupClassifier", index=0)
"""

from .types import TrainingMetadata
from .utils import timestamp_now
from .loaders import (
    MODEL_LOADER_REGISTRY,
    ModelLoader,
    get_loader,
    load_model_from_checkpoint,
    register_loader,
    register_model_loader,
)
from .managers import MetadataManager
from .utils.optuna import serialize_optuna_study


def save_model_and_metadata(
    model,
    metadata: TrainingMetadata,
    *,
    state_dict_only: bool = True,
):
    """
    Convenience wrapper to save a model (or its state_dict) plus TrainingMetadata.
    
    This mirrors the older functional API used in notebooks while delegating to
    the current MetadataManager under the hood.
    """
    manager = MetadataManager()
    return manager.save_model(model, metadata, state_dict_only=state_dict_only)


__all__ = [
    # Main API
    "MetadataManager",
    "save_model_and_metadata",
    # Core
    "TrainingMetadata",
    "timestamp_now",
    # Model loading (used by MetadataManager internally)
    "load_model_from_checkpoint",
    # Loader system
    "ModelLoader",
    "MODEL_LOADER_REGISTRY",
    "get_loader",
    "register_loader",
    "register_model_loader",
    # Optuna
    "serialize_optuna_study",
]
