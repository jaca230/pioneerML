"""
ZenML pipeline for training the endpoint regressor on raw hit graphs.
"""

import contextlib
import io
import os
import sys
import warnings
from typing import Any, Dict, Optional

import optuna
import pytorch_lightning as pl
import torch
import torch.nn as nn
from zenml import pipeline, step

from pioneerml.data import load_hits_and_info
from pioneerml.models.regressors.endpoint_regressor import EndpointRegressor
from pioneerml.training.datamodules import EndpointDataModule
from pioneerml.training.lightning import GraphLightningModule
from pioneerml.zenml.materializers import TorchTensorMaterializer
from pioneerml.zenml.utils import detect_available_accelerator


def _run_silently(fn):
    """Run a function while suppressing stdout/stderr and Lightning warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pl._logger.setLevel("ERROR")
        buffer_out, buffer_err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buffer_out), contextlib.redirect_stderr(buffer_err):
            return fn()


@step
def build_endpoint_datamodule(
    hits_pattern: Optional[str] = None,
    info_pattern: Optional[str] = None,
    max_files: Optional[int] = None,
    limit_groups: Optional[int] = None,
    min_hits: int = 2,
    batch_size: int = 32,
    num_workers: Optional[int] = None,  # None means auto-detect
    pin_memory: bool = False,
    val_split: float = 0.15,
    test_split: float = 0.0,
    seed: int = 42,
    num_quantiles: int = 3,
) -> EndpointDataModule:
    """Load data and build an EndpointDataModule in one step."""
    if hits_pattern is None or info_pattern is None:
        raise ValueError("hits_pattern and info_pattern are required but were not provided")

    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        num_workers = max(1, cpu_count - 1)
        print(f"Auto-detected num_workers: {num_workers} (from {cpu_count} CPU cores, using cores-1)", file=sys.stderr, flush=True)
    else:
        print(f"Using num_workers: {num_workers}", file=sys.stderr, flush=True)

    print(f"Starting to load data from: hits={hits_pattern}, info={info_pattern}", file=sys.stderr, flush=True)
    groups = load_hits_and_info(
        hits_pattern=hits_pattern,
        info_pattern=info_pattern,
        max_files=max_files,
        limit_groups=limit_groups,
        min_hits=min_hits,
        include_hit_labels=False,
        verbose=True,
    )
    print(f"Loaded {len(groups)} groups. Building datamodule...", file=sys.stderr, flush=True)
    datamodule = EndpointDataModule(
        records=groups,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
        num_quantiles=num_quantiles,
    )
    datamodule.setup(stage="fit")
    train_size = len(datamodule.train_dataset) if datamodule.train_dataset else 0
    val_size = len(datamodule.val_dataset) if datamodule.val_dataset else 0
    print(f"Setup complete. Train: {train_size}, Val: {val_size}", file=sys.stderr, flush=True)
    return datamodule


@step(enable_cache=False)
def run_endpoint_hparam_search(
    datamodule: EndpointDataModule,
    n_trials: int = 1,
    max_epochs: int = 1,
    limit_train_batches: int | float | None = 0.8,
    limit_val_batches: int | float | None = 1.0,
    storage: str | None = None,
    study_name: str = "endpoint_regressor",
) -> Dict[str, Any]:
    """Run Optuna hyperparameter search for endpoint regression."""
    datamodule.setup(stage="fit")
    accelerator, devices = detect_available_accelerator()

    def objective(trial: optuna.Trial) -> float:
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        hidden_raw = trial.suggest_categorical("hidden", [128, 160, 192])
        heads = trial.suggest_categorical("heads", [2, 4, 8])
        hidden = max(heads, (hidden_raw // heads) * heads)
        layers = trial.suggest_int("layers", 2, 4)
        dropout = trial.suggest_float("dropout", 0.05, 0.3)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        datamodule.batch_size = batch_size

        model = EndpointRegressor(
            in_channels=4,
            prob_dimension=0,
            hidden=hidden,
            heads=heads,
            layers=layers,
            dropout=dropout,
        )
        lightning_module = GraphLightningModule(
            model,
            task="regression",
            lr=lr,
            weight_decay=weight_decay,
            loss_fn=nn.MSELoss(),
        )

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

        trainer.fit(lightning_module, datamodule=datamodule)
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
def train_best_endpoint(
    best_params: Dict[str, Any],
    datamodule: EndpointDataModule,
    max_epochs: int = 5,
    early_stopping: bool = True,
    early_stopping_patience: int = 6,
    early_stopping_monitor: str = "val_loss",
    early_stopping_mode: str = "min",
) -> GraphLightningModule:
    """Train the final endpoint regressor using the best hyperparameters."""
    datamodule.setup(stage="fit")
    if "batch_size" in best_params:
        datamodule.batch_size = int(best_params["batch_size"])

    heads_val = int(best_params.get("heads", 4))
    hidden_raw = int(best_params.get("hidden", 160))
    hidden_val = max(heads_val, (hidden_raw // heads_val) * heads_val)

    model = EndpointRegressor(
        in_channels=4,
        prob_dimension=0,
        hidden=hidden_val,
        heads=heads_val,
        layers=int(best_params.get("layers", 3)),
        dropout=float(best_params.get("dropout", 0.1)),
    )
    lightning_module = GraphLightningModule(
        model,
        task="regression",
        lr=float(best_params.get("lr", 1e-3)),
        weight_decay=float(best_params.get("weight_decay", 1e-5)),
        loss_fn=nn.MSELoss(),
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

    _run_silently(lambda: trainer.fit(lightning_module, datamodule=datamodule))
    lightning_module.training_config = {
        "max_epochs": max_epochs,
        "early_stopping": early_stopping,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_monitor": early_stopping_monitor,
        "early_stopping_mode": early_stopping_mode,
    }
    lightning_module.final_epochs_run = int(getattr(trainer, "current_epoch", -1)) + 1
    return lightning_module.eval()


@step(
    enable_cache=False,
    output_materializers=(TorchTensorMaterializer, TorchTensorMaterializer),
)
def collect_endpoint_predictions(
    module: GraphLightningModule, datamodule: EndpointDataModule
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect predictions and targets from the validation set."""
    datamodule.setup(stage="fit")
    val_loader = datamodule.val_dataloader()
    if isinstance(val_loader, list) and len(val_loader) == 0:
        val_loader = datamodule.train_dataloader()

    preds, targets = [], []
    device = next(module.parameters()).device
    module.eval()
    for batch in val_loader:
        batch = batch.to(device)
        with torch.no_grad():
            out = module(batch).detach().cpu()
        target = batch.y.detach().cpu()

        if target.dim() == 1 and out.dim() == 2 and target.numel() % out.shape[-1] == 0:
            target = target.view(-1, out.shape[-1])
        elif target.dim() == 1:
            target = target.unsqueeze(0)

        preds.append(out)
        targets.append(target)

    preds = torch.cat(preds)
    targets = torch.cat(targets)
    # Flatten to 2D for generic regression diagnostics: [num_samples, features]
    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)
    return preds_flat, targets_flat


@pipeline
def endpoint_optuna_pipeline(
    build_datamodule_params: Optional[Dict[str, Any]] = None,
    run_hparam_search_params: Optional[Dict[str, Any]] = None,
    train_best_model_params: Optional[Dict[str, Any]] = None,
):
    """
    Complete pipeline for training endpoint regressor with Optuna hyperparameter tuning.
    """
    dm_kwargs = dict(build_datamodule_params or {})
    search_kwargs = dict(run_hparam_search_params or {})
    train_kwargs = dict(train_best_model_params or {})

    datamodule = (
        build_endpoint_datamodule.with_options(parameters=dm_kwargs)()
        if dm_kwargs
        else build_endpoint_datamodule()
    )
    best_params = (
        run_endpoint_hparam_search.with_options(parameters=search_kwargs)(datamodule)
        if search_kwargs
        else run_endpoint_hparam_search(datamodule)
    )
    trained_module = (
        train_best_endpoint.with_options(parameters=train_kwargs)(best_params, datamodule)
        if train_kwargs
        else train_best_endpoint(best_params, datamodule)
    )
    preds_and_targets = collect_endpoint_predictions(trained_module, datamodule)
    return trained_module, datamodule, preds_and_targets, best_params
