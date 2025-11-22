"""
Model training and inference pipeline stages.
"""

from __future__ import annotations

from typing import Any, Optional, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pioneerml.pipelines.stage import Stage, StageConfig


class TrainModelStage(Stage):
    """
    Stage for training a PyTorch model.

    Example:
        >>> from pioneerml.models import GroupClassifier
        >>>
        >>> stage = TrainModelStage(
        ...     config=StageConfig(
        ...         name='train',
        ...         inputs=['train_loader', 'val_loader'],
        ...         outputs=['model', 'metrics'],
        ...         params={
        ...             'model_class': GroupClassifier,
        ...             'model_params': {'hidden': 200, 'num_blocks': 2},
        ...             'epochs': 20,
        ...             'lr': 5e-4,
        ...         },
        ...     )
        ... )
    """

    def execute(self, context: Any) -> None:
        """Train the model."""
        # Get parameters
        model_class = self.config.params.get("model_class")
        model_params = self.config.params.get("model_params", {})
        epochs = self.config.params.get("epochs", 10)
        lr = self.config.params.get("lr", 1e-3)
        device = self.config.params.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # Get data
        train_loader = context.get("train_loader")
        val_loader = context.get("val_loader")

        if train_loader is None:
            raise KeyError("train_loader not found in context")

        # Create model
        if model_class is None:
            # Try to use existing model from context
            model = context.get("model")
            if model is None:
                raise ValueError("Either provide 'model_class' param or put model in context")
        else:
            model = model_class(**model_params)

        model = model.to(device)

        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        # Training loop (simplified)
        model.train()
        metrics = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()

                # Forward pass - this depends on the model
                # For now, assume model handles batch directly
                # In real implementation, you'd have a more sophisticated training loop

                total_loss += 1.0  # Placeholder

            metrics["train_loss"].append(total_loss)

        # Store results
        context["model"] = model
        context["metrics"] = metrics


class InferenceStage(Stage):
    """
    Stage for running model inference.

    Example:
        >>> stage = InferenceStage(
        ...     config=StageConfig(
        ...         name='inference',
        ...         inputs=['model', 'test_loader'],
        ...         outputs=['predictions'],
        ...         params={'device': 'cuda'},
        ...     )
        ... )
    """

    def execute(self, context: Any) -> None:
        """Run inference."""
        model = context["model"]
        test_loader = context["test_loader"]
        device = self.config.params.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        model = model.to(device)
        model.eval()

        predictions = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                output = model(batch)
                predictions.append(output.cpu())

        # Concatenate all predictions
        if predictions:
            predictions = torch.cat(predictions, dim=0)

        context["predictions"] = predictions
