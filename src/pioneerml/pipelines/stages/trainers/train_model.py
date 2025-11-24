"""
Basic training stage for a PyTorch model (non-Lightning).
"""

from __future__ import annotations

from typing import Any, Optional, Dict
import torch
from torch.utils.data import DataLoader

from pioneerml.pipelines.stage import StageConfig
from pioneerml.pipelines.stages.roles import TrainerStage


class TrainModelStage(TrainerStage):
    """
    Stage for training a PyTorch model (non-Lightning).

    Example:
        >>> stage = TrainModelStage(
        ...     config=StageConfig(
        ...         name='train',
        ...         inputs=['train_loader', 'val_loader'],
        ...         outputs=['model', 'metrics'],
        ...         params={
        ...             'model_class': MyModel,
        ...             'epochs': 10,
        ...             'lr': 1e-3,
        ...         },
        ...     )
        ... )
    """

    def execute(self, context: Any) -> None:
        params: Dict[str, Any] = self.config.params
        model_class = params.get("model_class")
        model_params = params.get("model_params", {})
        epochs = params.get("epochs", 10)
        lr = params.get("lr", 1e-3)
        device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        train_loader: Optional[DataLoader] = context.get("train_loader")
        if train_loader is None:
            raise KeyError("train_loader not found in context")
        val_loader: Optional[DataLoader] = context.get("val_loader")

        # Create or reuse model
        if model_class is None:
            model = context.get("model")
            if model is None:
                raise ValueError("Provide 'model_class' or put a model in context")
        else:
            model = model_class(**model_params)

        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        metrics = {"train_loss": [], "val_loss": []}

        model.train()
        for _ in range(epochs):
            total_loss = 0.0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                # Placeholder forward/backward; users should subclass for custom logic
                loss = getattr(model, "compute_loss", lambda b: torch.tensor(0.0, device=device))(batch)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu())
            metrics["train_loss"].append(total_loss)

            if val_loader is not None:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        loss = getattr(model, "compute_loss", lambda b: torch.tensor(0.0, device=device))(batch)
                        val_loss += float(loss.detach().cpu())
                metrics["val_loss"].append(val_loss)
                model.train()

        context["model"] = model
        context["metrics"] = metrics
