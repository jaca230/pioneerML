"""
Loader for EndpointRegressor models.
"""

from __future__ import annotations

import torch

from pioneerml.metadata.types import TrainingMetadata
from pioneerml.models.regressors import EndpointRegressor
from .base import ModelLoader
from .registry import register_model_loader


@register_model_loader("EndpointRegressor")
class EndpointRegressorLoader(ModelLoader):
    """Loader for EndpointRegressor models."""

    @property
    def model_type(self) -> str:
        return "EndpointRegressor"

    def load_model(
        self,
        metadata: TrainingMetadata,
        *,
        device: str | torch.device = "cpu",
    ) -> torch.nn.Module:
        """Load an EndpointRegressor from metadata."""
        defaults = {
            "hidden": 160,
            "heads": 4,
            "layers": 2,
            "dropout": 0.1,
            # Older checkpoints did not use group_probs; set default to 0 for compatibility.
            "prob_dimension": 0,
        }

        params = self.extract_architecture_params(metadata, defaults)

        hidden = int(params["hidden"])
        heads = int(params["heads"])
        hidden = max(heads, (hidden // heads) * heads)

        return EndpointRegressor(
            in_channels=4,
            prob_dimension=int(params.get("prob_dimension", 3)),
            hidden=hidden,
            heads=heads,
            layers=int(params["layers"]),
            dropout=float(params["dropout"]),
        )
