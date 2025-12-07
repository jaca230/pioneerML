"""
Registry for model loaders.

This module provides a registry pattern for model loaders, allowing
polymorphic loading of models without hardcoded conditionals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .base import ModelLoader
from ..types import TrainingMetadata

try:
    from pioneerml.zenml.utils import find_project_root  # type: ignore
except Exception:  # pragma: no cover
    def find_project_root(start=None):
        from pathlib import Path
        return Path(start or Path.cwd()).resolve()

# Global registry mapping model_type strings to loader classes
MODEL_LOADER_REGISTRY: dict[str, type[ModelLoader]] = {}


def register_loader(model_type: str, loader_class: type[ModelLoader]) -> None:
    """
    Register a model loader class for a given model type.
    
    Args:
        model_type: Model type string (e.g., "GroupClassifier")
        loader_class: Loader class that inherits from ModelLoader
    """
    MODEL_LOADER_REGISTRY[model_type] = loader_class


def register_model_loader(model_type: str):
    """
    Decorator to register a model loader class.
    
    Usage:
        @register_model_loader("MyModel")
        class MyModelLoader(ModelLoader):
            ...
    """
    def decorator(loader_class: type[ModelLoader]):
        register_loader(model_type, loader_class)
        return loader_class
    return decorator


def get_loader(model_type: str) -> ModelLoader:
    """
    Get a loader instance for the given model type.
    
    Args:
        model_type: Model type string
    
    Returns:
        Loader instance
    
    Raises:
        ValueError: If no loader is registered for the model type
    """
    if model_type not in MODEL_LOADER_REGISTRY:
        available = ", ".join(sorted(MODEL_LOADER_REGISTRY.keys()))
        raise ValueError(
            f"No loader registered for model_type='{model_type}'. "
            f"Available types: {available}"
        )
    
    loader_class = MODEL_LOADER_REGISTRY[model_type]
    return loader_class()


def load_model_from_checkpoint(
    model_type: str,
    checkpoint_path: Path | str | None = None,
    metadata: TrainingMetadata | None = None,
    *,
    root: Path | str | None = None,
    index: int = 0,
    device: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, TrainingMetadata]:
    """
    Load a model from a checkpoint, reconstructing architecture from metadata.
    
    This function uses the registered loaders to polymorphically load any model type.
    
    Args:
        model_type: Model type (e.g., "GroupClassifier", "GroupSplitter")
        checkpoint_path: Explicit checkpoint path (if None, will find using model_type and index)
        metadata: Explicit metadata (if None, will load from checkpoint path)
        root: Project root path (defaults to find_project_root())
        index: Index of checkpoint to select if checkpoint_path is None (0 = most recent)
        device: Device to load model onto
    
    Returns:
        Tuple of (model, metadata)
    """
    # Find checkpoint and metadata if not provided
    # Lazy import to avoid circular dependency
    if checkpoint_path is None:
        from ..managers.checkpoint_manager import CheckpointManager
        checkpoint_manager = CheckpointManager(root=root)
        checkpoint_path, metadata = checkpoint_manager.find(model_type, index=index)
        if checkpoint_path is None:
            raise ValueError(f"No checkpoint found for model_type={model_type}")
    
    checkpoint_path = Path(checkpoint_path)
    
    if metadata is None:
        from ..managers.artifact_manager import ArtifactManager
        artifact_manager = ArtifactManager(root=root)
        metadata_file = checkpoint_path.parent / checkpoint_path.name.replace("_state_dict.pt", "_metadata.json")
        if metadata_file.exists():
            metadata = artifact_manager.load_metadata(metadata_file)
        else:
            raise ValueError(f"No metadata found for checkpoint {checkpoint_path}")
    
    # Get the appropriate loader for this model type
    loader = get_loader(model_type)
    
    # Use the loader to instantiate the model
    device_obj = torch.device(device) if isinstance(device, str) else device
    model = loader.load_model(metadata, device=device_obj)
    
    # Load state dict
    state_dict = torch.load(checkpoint_path, map_location=device_obj)
    model.load_state_dict(state_dict)
    model = model.to(device_obj)
    model.eval()
    
    return model, metadata

