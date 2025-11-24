"""
Inference stage to run a trained model over a dataloader.
"""

from __future__ import annotations

from typing import Any
import torch

from pioneerml.pipelines.stage import StageConfig
from pioneerml.pipelines.stages.roles import RunnerStage


class InferenceStage(RunnerStage):
    """
    Stage for running model inference.

    Inputs:
        - 'model' (or override via params)
        - 'test_loader'
    Outputs:
        - 'predictions'
    """

    def execute(self, context: Any) -> None:
        model = context.get("model")
        if model is None:
            raise KeyError("model not found in context")
        test_loader = context.get("test_loader")
        if test_loader is None:
            raise KeyError("test_loader not found in context")

        device = self.config.params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        predictions = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                output = model(batch)
                predictions.append(output.cpu())

        if predictions:
            predictions = torch.cat(predictions, dim=0)
        context["predictions"] = predictions
