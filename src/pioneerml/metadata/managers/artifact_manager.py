"""
ArtifactManager: Handles saving and loading models with metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ..types import TrainingMetadata
from ..utils import timestamp_now

try:
    # Reuse project root helper if available
    from pioneerml.zenml.utils import find_project_root  # type: ignore
except Exception:  # pragma: no cover
    def find_project_root(start: Path | None = None) -> Path:
        return Path(start or Path.cwd()).resolve()


class ArtifactManager:
    """Manages model artifacts: saving and loading models with metadata."""
    
    def __init__(self, root: Path | str | None = None):
        """
        Initialize ArtifactManager.
        
        Args:
            root: Project root path (defaults to find_project_root())
        """
        self.root = Path(root or find_project_root())
    
    def build_paths(
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
        ts = timestamp or timestamp_now()
        root_path = self.root / "trained_models" / model_type.lower()
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
    
    def save(
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
        paths = self.build_paths(
            metadata.model_type, timestamp=metadata.timestamp, run_name=metadata.run_name
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
    
    def load_metadata(self, path: Path | str) -> TrainingMetadata:
        """
        Load TrainingMetadata from a JSON file.
        
        Args:
            path: Path to metadata JSON file
        
        Returns:
            TrainingMetadata object
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return TrainingMetadata.from_dict(data)

