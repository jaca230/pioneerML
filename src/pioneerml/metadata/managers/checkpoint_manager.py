"""
CheckpointManager: Handles finding, listing, and formatting checkpoint information.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..types import TrainingMetadata

try:
    # Reuse project root helper if available
    from pioneerml.zenml.utils import find_project_root  # type: ignore
except Exception:  # pragma: no cover
    def find_project_root(start: Path | None = None) -> Path:
        return Path(start or Path.cwd()).resolve()


class CheckpointManager:
    """Manages checkpoint discovery, listing, and formatting."""
    
    def __init__(self, root: Path | str | None = None, artifact_manager=None):
        """
        Initialize CheckpointManager.
        
        Args:
            root: Project root path (defaults to find_project_root())
            artifact_manager: ArtifactManager instance (will create if None)
        """
        self.root = Path(root or find_project_root())
        # Avoid circular import by accepting artifact_manager as parameter
        if artifact_manager is None:
            from ..managers.artifact_manager import ArtifactManager
            self._artifact_manager = ArtifactManager(root=root)
        else:
            self._artifact_manager = artifact_manager
    
    def find(
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
        checkpoints_dir = self.root / "trained_models" / model_type.lower()
        
        if not checkpoints_dir.exists():
            return None, None
        
        checkpoint_suffix = "_state_dict.pt"
        metadata_suffix = "_metadata.json"
        
        # Find all checkpoints
        checkpoint_files = sorted(
            checkpoints_dir.glob(f"{model_type.lower()}*{checkpoint_suffix}"),
            reverse=True
        )
        
        if not checkpoint_files or index >= len(checkpoint_files):
            return None, None
        
        selected_checkpoint = checkpoint_files[index]
        
        # Find matching metadata
        metadata_file = Path(str(selected_checkpoint).replace(checkpoint_suffix, metadata_suffix))
        metadata = None
        if metadata_file.exists():
            metadata = self._artifact_manager.load_metadata(metadata_file)
        
        return selected_checkpoint, metadata
    
    def list_all(
        self,
        model_type: str,
    ) -> list[dict[str, Any]]:
        """
        List all available checkpoints for a model type with their metadata.
        
        Args:
            model_type: Model type (e.g., "GroupClassifier", "GroupSplitter")
        
        Returns:
            List of dicts with keys: checkpoint_path, metadata, timestamp, run_name, architecture
        """
        checkpoints_dir = self.root / "trained_models" / model_type.lower()
        
        if not checkpoints_dir.exists():
            return []
        
        checkpoint_suffix = "_state_dict.pt"
        metadata_suffix = "_metadata.json"
        
        checkpoint_files = sorted(
            checkpoints_dir.glob(f"{model_type.lower()}*{checkpoint_suffix}"),
            reverse=True
        )
        
        results = []
        for ckpt_path in checkpoint_files:
            metadata_file = Path(str(ckpt_path).replace(checkpoint_suffix, metadata_suffix))
            metadata = None
            if metadata_file.exists():
                metadata = self._artifact_manager.load_metadata(metadata_file)
            
            result = {
                "checkpoint_path": ckpt_path,
                "metadata": metadata,
                "timestamp": metadata.timestamp if metadata else None,
                "run_name": metadata.run_name if metadata else None,
                "architecture": metadata.model_architecture if metadata else None,
            }
            results.append(result)
        
        return results
    
    def format_info(
        self,
        checkpoints: list[dict[str, Any]],
        *,
        show_architecture: bool = True,
        architecture_keys: list[str] | None = None,
    ) -> str:
        """
        Format checkpoint information as a human-readable string.
        
        Args:
            checkpoints: List of checkpoint dicts from list_all()
            show_architecture: Whether to show architecture details
            architecture_keys: List of architecture keys to display (defaults to common ones)
        
        Returns:
            Formatted string ready to print
        """
        if architecture_keys is None:
            architecture_keys = ["hidden", "heads", "layers", "dropout", "num_blocks"]
        
        lines = [f"Found {len(checkpoints)} checkpoint(s):"]
        
        for i, ckpt_info in enumerate(checkpoints, 1):
            ckpt_path = ckpt_info["checkpoint_path"]
            meta = ckpt_info["metadata"]
            
            lines.append(f"  {i}. {ckpt_path.name}")
            
            if meta:
                lines.append(f"     Timestamp:     {meta.timestamp}")
                lines.append(f"     Run:           {meta.run_name or 'unknown'}")
                
                if show_architecture:
                    arch = meta.model_architecture
                    if arch:
                        # Build architecture string from requested keys
                        arch_parts = []
                        for key in architecture_keys:
                            if key in arch:
                                arch_parts.append(f"{key}={arch[key]}")
                        
                        if arch_parts:
                            lines.append(f"     Architecture:  {', '.join(arch_parts)}")
            else:
                lines.append("     Metadata:      NOT FOUND")
        
        return "\n".join(lines)
    
    def print_all(
        self,
        model_type: str,
        *,
        show_architecture: bool = True,
        architecture_keys: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        List and print all checkpoints for a model type.
        
        Convenience method that combines list_all() and format_info().
        
        Args:
            model_type: Model type (e.g., "GroupClassifier", "GroupSplitter")
            show_architecture: Whether to show architecture details
            architecture_keys: List of architecture keys to display
        
        Returns:
            List of checkpoint dicts (same as list_all())
        """
        checkpoints = self.list_all(model_type)
        formatted = self.format_info(
            checkpoints,
            show_architecture=show_architecture,
            architecture_keys=architecture_keys,
        )
        print(formatted)
        return checkpoints
