"""
Loader for PionStopRegressor models.
"""

from __future__ import annotations

import torch

from pioneerml.metadata.types import TrainingMetadata
from pioneerml.models.regressors import PionStopRegressor
from .base import ModelLoader
from .registry import register_model_loader


@register_model_loader("PionStopRegressor")
class PionStopRegressorLoader(ModelLoader):
    """Loader for PionStopRegressor models."""
    
    @property
    def model_type(self) -> str:
        return "PionStopRegressor"
    
    def load_model(
        self,
        metadata: TrainingMetadata,
        *,
        device: str | torch.device = "cpu",
    ) -> torch.nn.Module:
        """Load a PionStopRegressor from metadata."""
        defaults = {
            "hidden": 128,
            "heads": 4,
            "layers": 3,
            "dropout": 0.1,
        }
        
        params = self.extract_architecture_params(metadata, defaults)
        
        # Ensure hidden is divisible by heads
        hidden = int(params["hidden"])
        heads = int(params["heads"])
        hidden = max(heads, (hidden // heads) * heads)
        
        return PionStopRegressor(
            in_channels=5,
            hidden=hidden,
            heads=heads,
            layers=int(params["layers"]),
            dropout=float(params["dropout"]),
        )

