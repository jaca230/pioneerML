"""
ZenML pipeline for training the group classifier on real time-group data.

This pipeline loads preprocessed time groups from .npy files, runs Optuna
hyperparameter search, trains the best model, and collects predictions for evaluation.
"""

import contextlib
import io
import os
import sys
import warnings
from typing import Any, Dict, Mapping, Optional

import optuna
import pytorch_lightning as pl
import torch
import torch.nn as nn
from zenml import pipeline, step

from pioneerml.data import load_preprocessed_time_groups, NUM_GROUP_CLASSES
from pioneerml.models.classifiers.group_classifier import GroupClassifier
from pioneerml.optuna import OptunaStudyManager
from pioneerml.training.datamodules import GroupClassificationDataModule
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


def _compute_pos_weight(datamodule: GroupClassificationDataModule) -> torch.Tensor:
    """
    Compute per-class positive weights to rebalance BCEWithLogitsLoss.
    """
    datamodule.setup(stage="fit")
    train_ds = datamodule.train_dataset
    if train_ds is None or len(train_ds) == 0:
        return torch.ones(NUM_GROUP_CLASSES)

    num_classes = NUM_GROUP_CLASSES
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


@step
def build_group_classification_datamodule(
    file_pattern: Optional[str] = None,
    max_files: Optional[int] = None,
    limit_groups: Optional[int] = None,
    min_hits: int = 2,
    min_hits_per_label: int = 2,
    batch_size: int = 128,
    num_workers: Optional[int] = None,  # None means auto-detect
    pin_memory: bool = False,
    val_split: float = 0.15,
    test_split: float = 0.0,
    seed: int = 42,
) -> GroupClassificationDataModule:
    """
    Load data and build a GroupClassificationDataModule in one step.
    
    This avoids materializing a large list of dictionaries between steps.
    
    Args:
        file_pattern: Glob pattern for data files (required, but can be passed via parameters)
        num_workers: Number of DataLoader workers. If None, auto-detects based on CPU count.
            Set to 0 to disable multiprocessing.
    """
    if file_pattern is None:
        raise ValueError("file_pattern is required but was not provided")
    
    # Auto-detect num_workers if not specified
    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        # Use all cores - 1 (leave one core free for system)
        num_workers = max(1, cpu_count - 1)
        print(f"Auto-detected num_workers: {num_workers} (from {cpu_count} CPU cores, using cores-1)", file=sys.stderr, flush=True)
    else:
        print(f"Using num_workers: {num_workers}", file=sys.stderr, flush=True)
    
    print(f"Starting to load data from: {file_pattern}", file=sys.stderr, flush=True)
    groups = load_preprocessed_time_groups(
        file_pattern,
        max_files=max_files,
        limit_groups=limit_groups,
        min_hits=min_hits,
        min_hits_per_label=min_hits_per_label,
        verbose=True,
    )
    print(f"Loaded {len(groups)} groups. Building datamodule...", file=sys.stderr, flush=True)
    datamodule = GroupClassificationDataModule(
        records=groups,
        num_classes=NUM_GROUP_CLASSES,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
    )
    print(f"Calling setup(stage='fit')...", file=sys.stderr, flush=True)
    datamodule.setup(stage="fit")
    train_size = len(datamodule.train_dataset) if datamodule.train_dataset else 0
    val_size = len(datamodule.val_dataset) if datamodule.val_dataset else 0
    print(f"Setup complete. Train: {train_size}, Val: {val_size}", file=sys.stderr, flush=True)
    return datamodule


@step(enable_cache=False)
def run_group_classification_hparam_search(
    datamodule: GroupClassificationDataModule,
    n_trials: int = 25,
    max_epochs: int = 20,
    limit_train_batches: int | float | None = 0.8,
    limit_val_batches: int | float | None = 1.0,
    storage: str | None = None,
    study_name: str = "group_classifier",
) -> Dict[str, Any]:
    """
    Run Optuna hyperparameter search for group classification.

    If `storage` is provided, the study will be persisted (or resumed) from that
    storage backend (e.g., sqlite:///.../optuna.db). Use n_trials=0 to skip new
    trials and reuse the existing best trial. If no prior trials exist and
    n_trials<=0, a small default hyperparameter set is returned without error.
    """
    datamodule.setup(stage="fit")
    accelerator, devices = detect_available_accelerator()
    
    # Log hardware information
    if accelerator == "tpu":
        print(f"Using TPU", file=sys.stderr, flush=True)
    elif accelerator == "gpu":
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
        print(f"Using GPU: {gpu_name}", file=sys.stderr, flush=True)
        print(f"CUDA devices: {devices}", file=sys.stderr, flush=True)
        # Enable Tensor Core optimization for NVIDIA GPUs
        try:
            torch.set_float32_matmul_precision('medium')
            print(f"Enabled Tensor Core optimization (medium precision)", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"Warning: Could not set Tensor Core precision: {e}", file=sys.stderr, flush=True)
    elif accelerator == "mps":
        print(f"Using Apple Silicon GPU (MPS)", file=sys.stderr, flush=True)
    else:
        print(f"Using CPU", file=sys.stderr, flush=True)
    
    train_size = len(datamodule.train_dataset) if datamodule.train_dataset else 0
    val_size = len(datamodule.val_dataset) if datamodule.val_dataset else 0
    print(f"Dataset sizes - Train: {train_size}, Val: {val_size}", file=sys.stderr, flush=True)
    print(f"Starting Optuna search with {n_trials} trials, {max_epochs} epochs per trial...", file=sys.stderr, flush=True)

    pos_weight = _compute_pos_weight(datamodule)

    existing_trials = 0  # set after study creation

    def objective(trial: optuna.Trial) -> float:
        # Show both this-run index and cumulative index
        this_run_idx = trial.number - existing_trials + 1
        total_idx = trial.number + 1
        total_planned = existing_trials + max(n_trials, 0)
        trial_num = this_run_idx  # For consistency with print statements
        print(
            f"Trial {this_run_idx}/{n_trials} (cumulative {total_idx}/{total_planned}) starting...",
            file=sys.stderr,
            flush=True,
        )
        
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        hidden = trial.suggest_categorical("hidden", [128, 192, 256])
        num_blocks = trial.suggest_int("num_blocks", 2, 4)
        dropout = trial.suggest_float("dropout", 0.05, 0.25)
        lr = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

        print(f"Trial {trial_num} params: batch_size={batch_size}, hidden={hidden}, num_blocks={num_blocks}, dropout={dropout:.4f}, lr={lr:.6f}, weight_decay={weight_decay:.6f}", file=sys.stderr, flush=True)

        datamodule.batch_size = batch_size

        model = GroupClassifier(
            num_classes=NUM_GROUP_CLASSES,
            hidden=hidden,
            num_blocks=num_blocks,
            dropout=dropout,
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
            enable_progress_bar=True,  # Enable progress bar for visibility
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            gradient_clip_val=2.0,
        )

        print(f"Trial {trial_num} training {max_epochs} epochs...", file=sys.stderr, flush=True)
        trainer.fit(lightning_module, datamodule=datamodule)
        
        print(f"Trial {trial_num} validating...", file=sys.stderr, flush=True)
        val_metrics = trainer.validate(lightning_module, datamodule=datamodule, verbose=False)

        score = 0.0
        if val_metrics and isinstance(val_metrics[0], dict):
            accuracy = val_metrics[0].get("val_accuracy")
            if accuracy is not None:
                score = float(accuracy)
            else:
                loss = val_metrics[0].get("val_loss")
                if loss is not None:
                    score = 1.0 / (1.0 + float(loss))
        
        print(f"Trial {trial_num} completed with score: {score:.6f}", file=sys.stderr, flush=True)
        return score

    manager = OptunaStudyManager(
        project_root=None,
        study_name=study_name,
        direction="maximize",
        storage=storage,
    )
    study, storage_used = manager.create_or_load()
    existing_trials = len(study.trials)
    if n_trials > 0:
        print(f"Starting Optuna study (storage={storage_used}, name={study_name})...", file=sys.stderr, flush=True)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        print(f"Optuna search complete! Best score: {study.best_value:.6f}", file=sys.stderr, flush=True)
        print(f"Best params: {study.best_params}", file=sys.stderr, flush=True)
    else:
        print(f"Skipping new trials; reusing existing study best (storage={storage_used}, name={study_name})", file=sys.stderr, flush=True)
        if study.best_trial is None:
            # No trials exist; fall back to a reasonable default without error
            print("No existing trials found. Returning default hyperparameters.", file=sys.stderr, flush=True)
            default_params = {
                "batch_size": 64,
                "hidden": 192,
                "num_blocks": 3,
                "dropout": 0.1,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "best_score": 0.0,
                "n_trials": 0,
            }
            return default_params
        else:
            print(f"Loaded prior study with {len(study.trials)} trials. Best score: {study.best_value:.6f}", file=sys.stderr, flush=True)

    best_params = dict(study.best_params)
    best_params["best_score"] = study.best_value
    best_params["n_trials"] = len(study.trials)
    # Note: trial_history removed to keep best_params lightweight
    return best_params


@step(enable_cache=False)
def train_best_group_classifier(
    best_params: Dict[str, Any],
    datamodule: GroupClassificationDataModule,
    max_epochs: int = 50,
    early_stopping: bool = True,
    early_stopping_patience: int = 6,
    early_stopping_monitor: str = "val_loss",
    early_stopping_mode: str = "min",
) -> GraphLightningModule:
    """Train the final group classifier using the best hyperparameters."""
    print("Starting final model training...", file=sys.stderr, flush=True)
    datamodule.setup(stage="fit")
    if "batch_size" in best_params:
        datamodule.batch_size = int(best_params["batch_size"])

    train_size = len(datamodule.train_dataset) if datamodule.train_dataset else 0
    val_size = len(datamodule.val_dataset) if datamodule.val_dataset else 0
    print(f"Dataset sizes - Train: {train_size}, Val: {val_size}", file=sys.stderr, flush=True)

    model = GroupClassifier(
        num_classes=NUM_GROUP_CLASSES,
        hidden=int(best_params.get("hidden", 192)),
        num_blocks=int(best_params.get("num_blocks", 3)),
        dropout=float(best_params.get("dropout", 0.1)),
    )
    lightning_module = GraphLightningModule(
        model,
        task="classification",
        lr=float(best_params.get("lr", 1e-3)),
        weight_decay=float(best_params.get("weight_decay", 1e-4)),
        loss_fn=nn.BCEWithLogitsLoss(pos_weight=_compute_pos_weight(datamodule)),
    )

    accelerator, devices = detect_available_accelerator()
    
    # Log hardware information
    if accelerator == "tpu":
        print("Using TPU", file=sys.stderr, flush=True)
    elif accelerator == "gpu":
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
        print(f"Using GPU: {gpu_name}", file=sys.stderr, flush=True)
        print(f"CUDA devices: {devices}", file=sys.stderr, flush=True)
        # Enable Tensor Core optimization for NVIDIA GPUs
        try:
            torch.set_float32_matmul_precision('medium')
            print("Enabled Tensor Core optimization (medium precision)", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"Warning: Could not set Tensor Core precision: {e}", file=sys.stderr, flush=True)
    elif accelerator == "mps":
        print("Using Apple Silicon GPU (MPS)", file=sys.stderr, flush=True)
    else:
        print("Using CPU", file=sys.stderr, flush=True)
    
    callbacks = []
    if early_stopping:
        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor=early_stopping_monitor,
                mode=early_stopping_mode,
                patience=early_stopping_patience,
                verbose=True,  # Enable verbose for early stopping
            )
        )

    min_epochs = max_epochs if not early_stopping else None
    
    print(f"Training configuration:", file=sys.stderr, flush=True)
    print(f"  max_epochs: {max_epochs}", file=sys.stderr, flush=True)
    print(f"  early_stopping: {early_stopping}", file=sys.stderr, flush=True)
    if early_stopping:
        print(f"  early_stopping_patience: {early_stopping_patience}", file=sys.stderr, flush=True)
    print(f"Starting training...", file=sys.stderr, flush=True)

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,  # Enable progress bar for visibility
        gradient_clip_val=2.0,
        callbacks=callbacks,
    )

    # Don't use _run_silently - we want to see progress
    trainer.fit(lightning_module, datamodule=datamodule)
    
    final_epoch = int(getattr(trainer, "current_epoch", -1)) + 1
    print(f"Training complete! Final epoch: {final_epoch}", file=sys.stderr, flush=True)
    
    lightning_module.training_config = {
        "max_epochs": max_epochs,
        "min_epochs": min_epochs,
        "early_stopping": early_stopping,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_monitor": early_stopping_monitor,
        "early_stopping_mode": early_stopping_mode,
    }
    lightning_module.final_epochs_run = final_epoch
    return lightning_module.eval()


@step(
    enable_cache=False,
    output_materializers=(TorchTensorMaterializer, TorchTensorMaterializer),
)
def collect_group_classification_predictions(
    module: GraphLightningModule, datamodule: GroupClassificationDataModule
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

    return torch.cat(preds), torch.cat(targets)


@pipeline
def group_classification_optuna_pipeline(
    build_datamodule_params: Optional[Mapping[str, Any]] = None,
    run_hparam_search_params: Optional[Mapping[str, Any]] = None,
    train_best_model_params: Optional[Mapping[str, Any]] = None,
):
    """
    Complete pipeline for training group classifier with Optuna hyperparameter tuning.
    
    Args:
        build_datamodule_params: Parameters for build_group_classification_datamodule
            (includes file_pattern and all data loading + datamodule params)
        run_hparam_search_params: Parameters for run_group_classification_hparam_search
        train_best_model_params: Parameters for train_best_group_classifier
    """
    dm_kwargs = dict(build_datamodule_params or {})
    search_kwargs = dict(run_hparam_search_params or {})
    train_kwargs = dict(train_best_model_params or {})

    # Pass overrides via ZenML step options to ensure they are honored.
    datamodule = (
        build_group_classification_datamodule.with_options(parameters=dm_kwargs)()
        if dm_kwargs
        else build_group_classification_datamodule()
    )
    best_params = (
        run_group_classification_hparam_search.with_options(parameters=search_kwargs)(datamodule)
        if search_kwargs
        else run_group_classification_hparam_search(datamodule)
    )
    trained_module = (
        train_best_group_classifier.with_options(parameters=train_kwargs)(best_params, datamodule)
        if train_kwargs
        else train_best_group_classifier(best_params, datamodule)
    )
    predictions, targets = collect_group_classification_predictions(trained_module, datamodule)
    return trained_module, datamodule, predictions, targets, best_params
