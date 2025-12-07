"""
Loader for GroupSplitter models.
"""

from __future__ import annotations

import torch

from pioneerml.metadata.types import TrainingMetadata
from pioneerml.models.classifiers import GroupSplitter
from .base import ModelLoader
from .registry import register_model_loader


@register_model_loader("GroupSplitter")
class GroupSplitterLoader(ModelLoader):
    """Loader for GroupSplitter models."""
    
    @property
    def model_type(self) -> str:
        return "GroupSplitter"
    
    def load_model(
        self,
        metadata: TrainingMetadata,
        *,
        device: str | torch.device = "cpu",
    ) -> torch.nn.Module:
        """Load a GroupSplitter from metadata."""
        defaults = {
            "hidden": 128,
            "heads": 4,
            "layers": 3,
            "dropout": 0.1,
            "num_classes": 3,
            "in_channels": 5,  # Default, could be 8 if use_group_probs=True
        }
        
        params = self.extract_architecture_params(metadata, defaults)
        
        # Ensure hidden is divisible by heads
        hidden = int(params["hidden"])
        heads = int(params["heads"])
        hidden = (hidden // heads) * heads
        
        return GroupSplitter(
            in_channels=int(params["in_channels"]),
            hidden=hidden,
            heads=heads,
            layers=int(params["layers"]),
            dropout=float(params["dropout"]),
            num_classes=int(params["num_classes"]),
        )

