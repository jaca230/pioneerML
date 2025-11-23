"""
Model training and inference pipeline stages.
"""

from __future__ import annotations

from typing import Any, Optional, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from pioneerml.pipelines.stage import Stage, StageConfig
from pioneerml.training import GraphLightningModule, GraphDataModule, set_tensor_core_precision, default_precision_for_accelerator


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


class LightningTrainStage(Stage):
    """
    Stage for training models with PyTorch Lightning.

    Supports either passing an existing LightningModule/DataModule via the
    pipeline context or constructing them from configuration.
    """

    def execute(self, context: Any) -> None:
        params = self.config.params

        # Resolve LightningModule
        module: Optional[pl.LightningModule] = params.get("module") or context.get("lightning_module")
        if module is None:
            model = params.get("model") or context.get("model")
            model_class = params.get("model_class")
            if model is None and model_class is not None:
                model_params = params.get("model_params", {})
                model = model_class(**model_params)

            module_class = params.get("module_class", GraphLightningModule)
            module_params = params.get("module_params", {})

            if model is None and module_class is GraphLightningModule:
                raise ValueError("Provide 'model' or 'model_class' to build a GraphLightningModule.")

            module = module_class(model=model, **module_params) if model is not None else module_class(**module_params)

        # Resolve DataModule
        datamodule: Optional[pl.LightningDataModule] = params.get("datamodule") or context.get("datamodule")
        if datamodule is None:
            datamodule_class = params.get("datamodule_class", GraphDataModule)
            dm_kwargs = params.get("datamodule_kwargs", {})

            dataset = params.get("dataset") or context.get("train_dataset") or context.get("dataset")
            val_dataset = params.get("val_dataset") or context.get("val_dataset")
            test_dataset = params.get("test_dataset") or context.get("test_dataset")

            if dataset is None and params.get("records") is None:
                raise ValueError("Provide a datamodule, dataset, or records to construct a DataModule.")

            if params.get("records") is not None:
                datamodule = datamodule_class(records=params["records"], **dm_kwargs)
            else:
                datamodule = datamodule_class(
                    dataset=dataset,
                    train_dataset=context.get("train_dataset"),
                    val_dataset=val_dataset,
                    test_dataset=test_dataset,
                    **dm_kwargs,
                )

        trainer_params: Dict[str, Any] = params.get("trainer_params", {})
        accel = trainer_params.get("accelerator", "auto")
        prec = trainer_params.get("precision") or default_precision_for_accelerator(accel)
        trainer_params["precision"] = prec
        if accel in {"cuda", "gpu", "auto"}:
            set_tensor_core_precision(params.get("matmul_precision", "medium"))
        trainer = pl.Trainer(**trainer_params)
        trainer.fit(module, datamodule=datamodule)

        context["lightning_module"] = module
        context["trainer"] = trainer
        context["model"] = getattr(module, "model", module)

        if trainer.logged_metrics:
            context["metrics"] = {
                key: value.item() if hasattr(value, "item") else value
                for key, value in trainer.logged_metrics.items()
            }
