"""
ZenML pipeline for training the EventBuilder affinity model on mixed events.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import warnings
from typing import Any, Dict, Optional, Sequence

import optuna
import pytorch_lightning as pl
import torch
from zenml import pipeline, step

from pioneerml.data.event_mixer import MixedEventDataset
from pioneerml.training.datamodules import EventBuilderDataModule
from pioneerml.training.lightning_event_builder import EventBuilderLightning
from pioneerml.models.components.event_builder import EventBuilder
from pioneerml.zenml.utils import detect_available_accelerator


def _run_silently(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pl._logger.setLevel("ERROR")
        buffer_out, buffer_err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buffer_out), contextlib.redirect_stderr(buffer_err):
            return fn()


@step
def build_event_builder_datamodule(
    mixed_paths: Sequence[str],
    val_split: float = 0.1,
    batch_size: int = 24,
    num_workers: int = 0,
    radius_z: float = 0.5,
    seed: int = 42,
) -> EventBuilderDataModule:
    if not mixed_paths:
        raise ValueError("mixed_paths must be provided for EventBuilder training.")
    dm = EventBuilderDataModule(
        mixed_paths=mixed_paths,
        val_split=val_split,
        batch_size=batch_size,
        num_workers=num_workers,
        radius_z=radius_z,
        seed=seed,
    )
    dm.setup(stage="fit")
    train_len = len(dm.train_dataset) if dm.train_dataset else 0
    val_len = len(dm.val_dataset) if dm.val_dataset else 0
    print(f"EventBuilder datamodule ready. Train events: {train_len}, Val events: {val_len}", file=sys.stderr, flush=True)
    return dm


@step(enable_cache=False)
def run_event_builder_hparam_search(
    datamodule: EventBuilderDataModule,
    n_trials: int = 5,
    max_epochs: int = 5,
    limit_train_batches: int | float | None = 0.8,
    limit_val_batches: int | float | None = 1.0,
    storage: str | None = None,
    study_name: str = "event_builder",
) -> Dict[str, Any]:
    accelerator, devices = detect_available_accelerator()

    def objective(trial: optuna.Trial) -> float:
        batch_size = trial.suggest_categorical("batch_size", [16, 24, 32])
        hidden = trial.suggest_categorical("hidden", [96, 128, 160])
        heads = trial.suggest_categorical("heads", [2, 4, 8])
        hidden = max(heads, (hidden // heads) * heads)
        layers = trial.suggest_int("layers", 2, 4)
        dropout = trial.suggest_float("dropout", 0.05, 0.3)
        lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        datamodule.batch_size = batch_size

        model = EventBuilder(
            in_channels=25,
            hidden=hidden,
            heads=heads,
            layers=layers,
            dropout=dropout,
        )
        lightning_module = EventBuilderLightning(model, lr=lr, weight_decay=weight_decay)

        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            max_epochs=max_epochs,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
        )

        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        if isinstance(val_loader, list) and len(val_loader) == 0:
            val_loader = None

        trainer.fit(lightning_module, train_loader, val_loader)
        val_metrics = trainer.callback_metrics
        val_loss = val_metrics.get("val_loss")
        return float(val_loss.cpu().item()) if val_loss is not None else float("inf")

    study = optuna.create_study(direction="minimize", storage=storage, study_name=study_name, load_if_exists=True)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best_params = dict(study.best_params)
    best_params["best_score"] = study.best_value
    best_params["n_trials"] = len(study.trials)
    return best_params


@step(enable_cache=False)
def train_best_event_builder(
    best_params: Dict[str, Any],
    datamodule: EventBuilderDataModule,
    max_epochs: int = 10,
    early_stopping: bool = True,
    early_stopping_patience: int = 4,
    early_stopping_monitor: str = "val_loss",
    early_stopping_mode: str = "min",
) -> EventBuilderLightning:
    datamodule.setup(stage="fit")
    if "batch_size" in best_params:
        datamodule.batch_size = int(best_params["batch_size"])

    heads_val = int(best_params.get("heads", 4))
    hidden_raw = int(best_params.get("hidden", 128))
    hidden_val = max(heads_val, (hidden_raw // heads_val) * heads_val)

    model = EventBuilder(
        in_channels=25,
        hidden=hidden_val,
        heads=heads_val,
        layers=int(best_params.get("layers", 3)),
        dropout=float(best_params.get("dropout", 0.1)),
    )
    lightning_module = EventBuilderLightning(
        model,
        lr=float(best_params.get("lr", 3e-4)),
        weight_decay=float(best_params.get("weight_decay", 1e-4)),
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

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        callbacks=callbacks,
    )

    _run_silently(lambda: trainer.fit(lightning_module, datamodule.train_dataloader(), datamodule.val_dataloader()))
    lightning_module.final_epochs_run = int(getattr(trainer, "current_epoch", -1)) + 1
    return lightning_module.eval()


@pipeline
def event_builder_optuna_pipeline(
    build_datamodule_params: Optional[Dict[str, Any]] = None,
    run_hparam_search_params: Optional[Dict[str, Any]] = None,
    train_best_model_params: Optional[Dict[str, Any]] = None,
):
    dm_kwargs = dict(build_datamodule_params or {})
    search_kwargs = dict(run_hparam_search_params or {})
    train_kwargs = dict(train_best_model_params or {})

    datamodule = build_event_builder_datamodule.with_options(parameters=dm_kwargs)()
    best_params = run_event_builder_hparam_search.with_options(parameters=search_kwargs)(datamodule)
    trained_module = train_best_event_builder.with_options(parameters=train_kwargs)(best_params, datamodule)
    return trained_module, datamodule, best_params
