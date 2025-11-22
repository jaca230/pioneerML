"""
Evaluation and metrics pipeline stages.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional
import torch

from pioneerml.pipelines.stage import Stage, StageConfig


class EvaluateStage(Stage):
    """
    Stage for evaluating model predictions.

    Example:
        >>> def compute_metrics(predictions, targets):
        ...     accuracy = (predictions.argmax(1) == targets).float().mean()
        ...     return {'accuracy': accuracy.item()}
        ...
        >>> stage = EvaluateStage(
        ...     config=StageConfig(
        ...         name='evaluate',
        ...         inputs=['predictions', 'targets'],
        ...         outputs=['evaluation_metrics'],
        ...         params={'metric_fn': compute_metrics},
        ...     )
        ... )
    """

    def execute(self, context: Any) -> None:
        """Evaluate predictions."""
        # Get metric function
        metric_fn = self.config.params.get("metric_fn")

        if metric_fn is None:
            # Use default metrics
            metric_fn = self._default_metrics

        # Get predictions and targets
        predictions = context.get(self.inputs[0] if len(self.inputs) > 0 else "predictions")
        targets = context.get(self.inputs[1] if len(self.inputs) > 1 else "targets")

        if predictions is None:
            raise KeyError(f"'{self.inputs[0]}' not found in context")

        # Compute metrics
        if targets is not None:
            metrics = metric_fn(predictions, targets)
        else:
            # No targets available - just store predictions
            metrics = {"num_predictions": len(predictions)}

        # Store results
        output_key = self.outputs[0] if self.outputs else "evaluation_metrics"
        context[output_key] = metrics

    @staticmethod
    def _default_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute default metrics.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.

        Returns:
            Dictionary of metrics.
        """
        metrics = {}

        # For classification (multi-label binary)
        if predictions.dim() == 2 and targets.dim() == 2:
            # Assume binary classification with sigmoid outputs
            pred_binary = (predictions > 0.5).float()
            accuracy = (pred_binary == targets).float().mean()
            metrics["accuracy"] = accuracy.item()

        # For regression
        elif predictions.dim() <= 2 and targets.dim() <= 2:
            mse = ((predictions - targets) ** 2).mean()
            metrics["mse"] = mse.item()

        return metrics
