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
        checkpoint_path: str | None = None,
        state_dict: dict | None = None,
    ) -> torch.nn.Module:
        """Load an EndpointRegressor from metadata."""
        defaults = {
            "hidden": 160,
            "heads": 4,
            "layers": 2,
            "dropout": 0.1,
            # Older checkpoints may not use group_probs; default to 0, but override from metadata when available.
            "prob_dimension": 0,
        }

        # Prefer explicit prob_dimension from metadata if present
        if metadata.model_architecture:
            defaults["prob_dimension"] = int(metadata.model_architecture.get("prob_dimension", defaults["prob_dimension"]))
        if metadata.best_hyperparameters:
            defaults["prob_dimension"] = int(metadata.best_hyperparameters.get("prob_dimension", defaults["prob_dimension"]))

        # Infer prob_dimension from checkpoint weights if not specified
        if defaults["prob_dimension"] == 0:
            if state_dict is None and checkpoint_path is not None:
                try:
                    state_dict = torch.load(checkpoint_path, map_location=device)
                except Exception:
                    state_dict = None
            if state_dict is not None:
                w = state_dict.get("hit_encoder.feature_proj.weight")
                if w is not None and w.ndim == 2:
                    defaults["prob_dimension"] = max(0, w.shape[1] - 3)

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
