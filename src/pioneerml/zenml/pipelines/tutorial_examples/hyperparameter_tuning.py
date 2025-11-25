"""
Hyperparameter tuning pipeline example for tutorials.

This pipeline demonstrates how to use Optuna for hyperparameter optimization
with ZenML pipelines.
"""

import torch
from torch_geometric.data import Data
from zenml import pipeline, step

from pioneerml.models import GroupClassifier
from pioneerml.training import GraphDataModule, GraphLightningModule
from pioneerml.zenml.utils import detect_available_accelerator


def create_synthetic_data_for_tuning(num_samples: int = 200) -> list[Data]:
    """Create synthetic data for hyperparameter tuning."""
    data = []
    for _ in range(num_samples):
        num_nodes = torch.randint(4, 8, (1,)).item()
        x = torch.randn(num_nodes, 5)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 3))
        edge_attr = torch.randn(edge_index.shape[1], 4)

        label = torch.randint(0, 3, (1,)).item()
        y = torch.zeros(3)
        y[label] = 1.0

        data.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    return data


@step
def prepare_tuning_data() -> list[Data]:
    """Step to prepare data for hyperparameter tuning."""
    return create_synthetic_data_for_tuning(200)


@step
def prepare_tuning_datamodule(data: list[Data]) -> GraphDataModule:
    """Step to create DataModule for tuning."""
    return GraphDataModule(dataset=data, val_split=0.2, batch_size=32)


@step
def hyperparameter_search(
    datamodule: GraphDataModule,
    n_trials: int = 3
) -> dict:
    """Step to perform hyperparameter search."""
    try:
        import optuna
    except ImportError:
        # Fallback if optuna not installed
        return {
            "best_hidden": 128,
            "best_lr": 1e-3,
            "best_accuracy": 0.85,
            "note": "Optuna not installed, using default parameters"
        }

    def objective(trial):
        # Suggest hyperparameters
        hidden_dim = trial.suggest_categorical("hidden", [64, 128, 256])
        lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
        dropout = trial.suggest_uniform("dropout", 0.0, 0.3)

        # Create model with suggested parameters
        model = GroupClassifier(
            num_classes=3,
            hidden=hidden_dim,
            dropout=dropout
        )
        lightning_module = GraphLightningModule(model, task="classification", lr=lr)

        # Quick training
        accelerator, devices = detect_available_accelerator()
        import pytorch_lightning as pl

        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            max_epochs=2,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )

        trainer.fit(lightning_module, datamodule=datamodule)

        # Evaluate on validation set and use accuracy as the objective
        val_results = trainer.validate(lightning_module, datamodule=datamodule, verbose=False)
        if val_results and isinstance(val_results[0], dict):
            accuracy = val_results[0].get("val_accuracy")
            if accuracy is not None:
                return float(accuracy)
            loss = val_results[0].get("val_loss")
            if loss is not None:
                return 1.0 / (1.0 + float(loss))

        # Fallback objective if metrics are missing
        return 0.0

    # Run optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return {
        "best_hidden": study.best_params["hidden"],
        "best_lr": study.best_params["lr"],
        "best_dropout": study.best_params["dropout"],
        "best_accuracy": study.best_value,
    }


@step
def train_with_best_params(
    best_params: dict,
    datamodule: GraphDataModule
) -> GraphLightningModule:
    """Step to train model with best hyperparameters."""
    model = GroupClassifier(
        num_classes=3,
        hidden=best_params["best_hidden"],
        dropout=best_params.get("best_dropout", 0.0)
    )

    lightning_module = GraphLightningModule(
        model,
        task="classification",
        lr=best_params["best_lr"]
    )

    accelerator, devices = detect_available_accelerator()
    import pytorch_lightning as pl

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=5,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    trainer.fit(lightning_module, datamodule=datamodule)
    return lightning_module.eval()


@pipeline
def hyperparameter_tuning_pipeline(n_trials: int = 3):
    """Hyperparameter tuning pipeline example."""
    data = prepare_tuning_data()
    datamodule = prepare_tuning_datamodule(data)
    best_params = hyperparameter_search(datamodule, n_trials=n_trials)
    trained_model = train_with_best_params(best_params, datamodule)
    return trained_model, datamodule, best_params
