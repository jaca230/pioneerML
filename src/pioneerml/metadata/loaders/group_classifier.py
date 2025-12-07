"""
Loader for GroupClassifier models.
"""

from __future__ import annotations

import torch

from pioneerml.metadata.types import TrainingMetadata
from pioneerml.models.classifiers import GroupClassifier
from .base import ModelLoader
from .registry import register_model_loader


@register_model_loader("GroupClassifier")
class GroupClassifierLoader(ModelLoader):
    """Loader for GroupClassifier models."""
    
    @property
    def model_type(self) -> str:
        return "GroupClassifier"
    
    def load_model(
        self,
        metadata: TrainingMetadata,
        *,
        device: str | torch.device = "cpu",
    ) -> torch.nn.Module:
        """Load a GroupClassifier from metadata."""
        defaults = {
            "hidden": 192,
            "num_blocks": 3,
            "dropout": 0.1,
            "num_classes": 3,
        }
        
        params = self.extract_architecture_params(metadata, defaults)
        
        return GroupClassifier(
            hidden=int(params["hidden"]),
            num_blocks=int(params["num_blocks"]),
            dropout=float(params["dropout"]),
            num_classes=int(params["num_classes"]),
        )

