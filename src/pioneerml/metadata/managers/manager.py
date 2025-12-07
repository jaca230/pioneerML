"""
MetadataManager: Main class for managing model checkpoints, artifacts, and metadata.

This is the primary entry point for all metadata operations. It uses helper
classes (CheckpointManager, ArtifactManager) to organize functionality.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..types import TrainingMetadata
from ..loaders.registry import load_model_from_checkpoint as _load_model_from_checkpoint
from .artifact_manager import ArtifactManager
from .checkpoint_manager import CheckpointManager

try:
    # Reuse project root helper if available
    from pioneerml.zenml.utils import find_project_root  # type: ignore
except Exception:  # pragma: no cover
    def find_project_root(start: Path | None = None) -> Path:
        return Path(start or Path.cwd()).resolve()


class MetadataManager:
    """
    Main manager for model metadata, checkpoints, and artifacts.
    
    This class provides a unified interface for:
    - Finding and listing checkpoints
    - Loading models from checkpoints
    - Saving models and metadata
    - Formatting checkpoint information
    
    It uses helper classes internally:
    - checkpoints: CheckpointManager for checkpoint operations
    - artifacts: ArtifactManager for saving/loading artifacts
    """
    
    def __init__(self, root: Path | str | None = None):
        """
        Initialize MetadataManager.
        
        Args:
            root: Project root path (defaults to find_project_root())
        """
        self.root = Path(root or find_project_root())
        self.artifacts = ArtifactManager(root=self.root)
        self.checkpoints = CheckpointManager(root=self.root, artifact_manager=self.artifacts)
    
    # Checkpoint operations (delegated to CheckpointManager)
    
    def find_checkpoint(
        self,
        model_type: str,
        *,
        index: int = 0,
    ) -> tuple[Path | None, TrainingMetadata | None]:
        """
        Find a checkpoint for a given model type.
        
        Args:
            model_type: Model type (e.g., "GroupClassifier", "GroupSplitter")
            index: Index of checkpoint to select (0 = most recent, 1 = second most recent, etc.)
        
        Returns:
            Tuple of (checkpoint_path, metadata) or (None, None) if not found
        """
        return self.checkpoints.find(model_type, index=index)
    
    def list_checkpoints(
        self,
        model_type: str,
    ) -> list[dict]:
        """
        List all available checkpoints for a model type with their metadata.
        
        Args:
            model_type: Model type (e.g., "GroupClassifier", "GroupSplitter")
        
        Returns:
            List of dicts with keys: checkpoint_path, metadata, timestamp, run_name, architecture
        """
        return self.checkpoints.list_all(model_type)
    
    def print_checkpoints(
        self,
        model_type: str,
        *,
        show_architecture: bool = True,
        architecture_keys: list[str] | None = None,
    ) -> list[dict]:
        """
        List and print all checkpoints for a model type.
        
        Args:
            model_type: Model type (e.g., "GroupClassifier", "GroupSplitter")
            show_architecture: Whether to show architecture details
            architecture_keys: List of architecture keys to display
        
        Returns:
            List of checkpoint dicts
        """
        return self.checkpoints.print_all(
            model_type,
            show_architecture=show_architecture,
            architecture_keys=architecture_keys,
        )
    
    # Model loading operations
    
    def load_model(
        self,
        model_type: str,
        checkpoint_path: Path | str | None = None,
        metadata: TrainingMetadata | None = None,
        *,
        index: int = 0,
        device: str | torch.device = "cpu",
    ) -> tuple[torch.nn.Module, TrainingMetadata]:
        """
        Load a model from a checkpoint using the registered ModelLoader for its type.
        
        Args:
            model_type: Model type (e.g., "GroupClassifier", "GroupSplitter")
            checkpoint_path: Explicit checkpoint path (if None, will find using model_type and index)
            metadata: Explicit metadata (if None, will load from checkpoint path)
            index: Index of checkpoint to select if checkpoint_path is None (0 = most recent)
            device: Device to load model onto
        
        Returns:
            Tuple of (model, metadata)
        """
        return _load_model_from_checkpoint(
            model_type,
            checkpoint_path=checkpoint_path,
            metadata=metadata,
            root=self.root,
            index=index,
            device=device,
        )
    
    # Artifact operations (delegated to ArtifactManager)
    
    def save_model(
        self,
        model: torch.nn.Module | None,
        metadata: TrainingMetadata,
        *,
        state_dict_only: bool = True,
    ) -> dict[str, Path]:
        """
        Save a model (state_dict or full checkpoint) plus standardized metadata JSON.
        
        Args:
            model: PyTorch model to save (can be None to save only metadata)
            metadata: TrainingMetadata object
            state_dict_only: If True, save only state_dict; if False, save full model
        
        Returns:
            Dictionary of artifact paths used
        """
        return self.artifacts.save(model, metadata, state_dict_only=state_dict_only)
    
    def load_metadata(self, path: Path | str) -> TrainingMetadata:
        """
        Load TrainingMetadata from a JSON file.
        
        Args:
            path: Path to metadata JSON file
        
        Returns:
            TrainingMetadata object
        """
        return self.artifacts.load_metadata(path)
    
    def build_artifact_paths(
        self,
        model_type: str,
        *,
        timestamp: str | None = None,
        run_name: str | None = None,
    ) -> dict[str, Path]:
        """
        Build standard artifact paths under trained_models/<model_type>/.
        
        Args:
            model_type: Model type (e.g., "GroupClassifier")
            timestamp: Timestamp string (defaults to current time)
            run_name: Optional run name to include in filename
        
        Returns:
            Dictionary with keys: dir, state_dict, metadata, full_checkpoint
        """
        return self.artifacts.build_paths(model_type, timestamp=timestamp, run_name=run_name)

