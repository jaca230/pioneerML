"""
PyTorch Lightning training stage for graph models.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import pytorch_lightning as pl

from pioneerml.training import GraphLightningModule, GraphDataModule, set_tensor_core_precision, default_precision_for_accelerator
from pioneerml.pipelines.stage import StageConfig
from pioneerml.pipelines.stages.roles import TrainerStage


class LightningTrainStage(TrainerStage):
    """
    Stage for training models with PyTorch Lightning.

    Supports passing an existing LightningModule/DataModule via the context or constructing them from configuration.
    """

    def execute(self, context: Any) -> None:
        params = self.config.params

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
        context["datamodule"] = datamodule

        if trainer.logged_metrics:
            context["metrics"] = {
                key: value.item() if hasattr(value, "item") else value
                for key, value in trainer.logged_metrics.items()
            }
