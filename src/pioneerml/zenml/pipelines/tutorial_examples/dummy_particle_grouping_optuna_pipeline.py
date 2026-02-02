"""
Optuna-tuned variant of the dummy particle grouping pipeline.

Runs a small Optuna sweep over model and training hyperparameters, then
trains a final model using the best configuration and collects predictions.
"""

import contextlib
import io
import warnings
from typing import Any, Dict, Mapping, Optional

import optuna
import pytorch_lightning as pl
import torch
import torch.nn as nn
from zenml import pipeline, step

from pioneerml.models.classifiers.group_classifier import GroupClassifier
from pioneerml.training.lightning import GraphLightningModule
from pioneerml.training.datamodules import GroupClassificationDataModule
from pioneerml.zenml.utils import detect_available_accelerator
from pioneerml.zenml.pipelines.tutorial_examples.dummy_particle_grouping_pipeline import (
    build_dummy_datamodule,
    collect_dummy_predictions,
)


def _run_silently(fn):
    """Run a function while suppressing stdout/stderr and Lightning warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pl._logger.setLevel("ERROR")
        buffer_out, buffer_err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buffer_out), contextlib.redirect_stderr(buffer_err):
            return fn()


def _compute_pos_weight(datamodule: GroupClassificationDataModule) -> torch.Tensor:
    """
    Compute per-class positive weights to rebalance BCEWithLogitsLoss.

    This does not change the dataset; it only re-weights the loss to pay
    more attention to under-represented labels (notably energy_high/low).
    """
    datamodule.setup(stage="fit")
    train_ds = datamodule.train_dataset
    if train_ds is None or len(train_ds) == 0:
        return torch.ones(1)

    num_classes = int(train_ds[0].y.numel())
    pos = torch.zeros(num_classes)
    total = 0
    for sample in train_ds:
        y = sample.y.float()
        pos += y
        total += 1

    pos = pos.clamp(min=1.0)
    neg = float(total) - pos
    pos_weight = neg / pos
    return pos_weight


@step(enable_cache=False)
def run_dummy_hparam_search(
    datamodule: GroupClassificationDataModule,
    n_trials: int = 6,
    max_epochs: int = 8,
    limit_train_batches: int | float | None = 0.7,
    limit_val_batches: int | float | None = 1.0,
) -> Dict[str, Any]:
    """
    Run a lightweight Optuna sweep over model/training hyperparameters.

    The search keeps epochs and batch limits modest to stay fast while still
    steering the model toward better configurations.
    """
    datamodule.setup(stage="fit")
    accelerator, devices = detect_available_accelerator()

    pos_weight = _compute_pos_weight(datamodule)

    def objective(trial: optuna.Trial) -> float:
        # Tune a handful of impactful parameters
        batch_size = trial.suggest_categorical("batch_size", [32, 48, 64])
        hidden = trial.suggest_categorical("hidden", [128, 192, 256])
        num_blocks = trial.suggest_int("num_blocks", 2, 4)
        dropout = trial.suggest_float("dropout", 0.05, 0.25)
        lr = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 2e-3, log=True)

        datamodule.batch_size = batch_size

        model = GroupClassifier(
            in_dim=4,
            edge_dim=4,
            hidden=hidden,
            heads=4,
            num_blocks=num_blocks,
            dropout=dropout,
            num_classes=datamodule.train_dataset[0].y.numel() if datamodule.train_dataset else 6,
        )
        lightning_module = GraphLightningModule(
            model,
            task="classification",
            lr=lr,
            weight_decay=weight_decay,
            loss_fn=nn.BCEWithLogitsLoss(pos_weight=pos_weight),
        )

        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            max_epochs=max_epochs,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            gradient_clip_val=1.0,
        )

        def fit():
            trainer.fit(lightning_module, datamodule=datamodule)

        def validate():
            return trainer.validate(lightning_module, datamodule=datamodule, verbose=False)

        _run_silently(fit)
        val_metrics = _run_silently(validate)

        if val_metrics and isinstance(val_metrics[0], dict):
            accuracy = val_metrics[0].get("val_accuracy")
            if accuracy is not None:
                return float(accuracy)
            loss = val_metrics[0].get("val_loss")
            if loss is not None:
                return 1.0 / (1.0 + float(loss))
        return 0.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_params["best_score"] = study.best_value
    best_params["n_trials"] = len(study.trials)
    return best_params


@step(enable_cache=False)
def train_best_dummy_model(
    best_params: Dict[str, Any],
    datamodule: GroupClassificationDataModule,
    max_epochs: int = 35,
    early_stopping: bool = True,
    early_stopping_patience: int = 6,
    early_stopping_monitor: str = "val_loss",
    early_stopping_mode: str = "min",
) -> GraphLightningModule:
    """Train a final model using the best hyperparameters from Optuna."""
    datamodule.setup(stage="fit")
    if "batch_size" in best_params:
        datamodule.batch_size = int(best_params["batch_size"])

    model = GroupClassifier(
        in_dim=4,
        edge_dim=4,
        hidden=int(best_params.get("hidden", 192)),
        heads=4,
        num_blocks=int(best_params.get("num_blocks", 3)),
        dropout=float(best_params.get("dropout", 0.1)),
        num_classes=datamodule.train_dataset[0].y.numel() if datamodule.train_dataset else 6,
    )
    lightning_module = GraphLightningModule(
        model,
        task="classification",
        lr=float(best_params.get("lr", 1e-3)),
        weight_decay=float(best_params.get("weight_decay", 1e-3)),
        loss_fn=nn.BCEWithLogitsLoss(pos_weight=_compute_pos_weight(datamodule)),
    )

    accelerator, devices = detect_available_accelerator()
    callbacks = []
    if early_stopping:
        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor=early_stopping_monitor,
                mode=early_stopping_mode,
                patience=early_stopping_patience,
                verbose=False,
            )
        )

    # If early stopping is disabled, force full training by setting min_epochs to max_epochs.
    min_epochs = max_epochs if not early_stopping else None

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        gradient_clip_val=1.0,
        callbacks=callbacks,
    )

    def fit():
        trainer.fit(lightning_module, datamodule=datamodule)

    _run_silently(fit)
    # Expose training meta for downstream inspection/plotting
    lightning_module.training_config = {
        "max_epochs": max_epochs,
        "min_epochs": min_epochs,
        "early_stopping": early_stopping,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_monitor": early_stopping_monitor,
        "early_stopping_mode": early_stopping_mode,
    }
    lightning_module.final_epochs_run = int(getattr(trainer, "current_epoch", -1)) + 1
    return lightning_module.eval()


@pipeline
def dummy_particle_grouping_optuna_pipeline(
    build_dummy_datamodule_params: Optional[Mapping[str, Any]] = None,
    run_dummy_hparam_search_params: Optional[Mapping[str, Any]] = None,
    train_best_dummy_model_params: Optional[Mapping[str, Any]] = None,
):
    """
    Pipeline wrapper that forwards JSON-serializable configs into each step.

    Args:
        build_dummy_datamodule_params: dict of keyword args for build_dummy_datamodule.
        run_dummy_hparam_search_params: dict of keyword args for run_dummy_hparam_search.
        train_best_dummy_model_params: dict of keyword args for train_best_dummy_model.
    """
    dm_kwargs = dict(build_dummy_datamodule_params or {})
    search_kwargs = dict(run_dummy_hparam_search_params or {})
    train_kwargs = dict(train_best_dummy_model_params or {})

    # Pass overrides via ZenML step options to ensure they are honored.
    datamodule = (
        build_dummy_datamodule.with_options(parameters=dm_kwargs)()
        if dm_kwargs
        else build_dummy_datamodule()
    )
    best_params = (
        run_dummy_hparam_search.with_options(parameters=search_kwargs)(datamodule)
        if search_kwargs
        else run_dummy_hparam_search(datamodule)
    )
    tuned_module = (
        train_best_dummy_model.with_options(parameters=train_kwargs)(best_params, datamodule)
        if train_kwargs
        else train_best_dummy_model(best_params, datamodule)
    )
    predictions, targets = collect_dummy_predictions(tuned_module, datamodule)
    return tuned_module, datamodule, predictions, targets, best_params
