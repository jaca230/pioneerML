"""
ZenML pipeline for training the group splitter on real time-group data.

This pipeline loads preprocessed splitter groups from .npy files, runs Optuna
hyperparameter search, trains the best model, and collects predictions for evaluation.
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

from pioneerml.data import load_splitter_groups, NUM_NODE_CLASSES
from pioneerml.models.classifiers.group_splitter import GroupSplitter
from pioneerml.training.datamodules import SplitterDataModule
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


def _compute_pos_weight(datamodule: SplitterDataModule) -> torch.Tensor:
    """
    Compute per-class positive weights to rebalance BCEWithLogitsLoss for splitter.
    """
    datamodule.setup(stage="fit")
    train_ds = datamodule.train_dataset
    if train_ds is None or len(train_ds) == 0:
        return torch.ones(NUM_NODE_CLASSES)

    num_classes = NUM_NODE_CLASSES
    pos = torch.zeros(num_classes)
    total_hits = 0
    for sample in train_ds:
        y = sample.y.float()  # [N, 3]
        pos += y.sum(dim=0)  # Sum over hits
        total_hits += y.shape[0]

    pos = pos.clamp(min=1.0)
    neg = float(total_hits) - pos
    pos_weight = neg / pos
    return pos_weight


@step
def build_splitter_datamodule(
    *,
    file_pattern: Optional[str] = None,
    max_files: Optional[int] = None,
    limit_groups: Optional[int] = None,
    min_hits: int = 3,
    use_group_probs: bool = False,
    batch_size: int = 8,
    num_workers: Optional[int] = None,  # None means auto-detect
    pin_memory: bool = False,
    val_split: float = 0.15,
    test_split: float = 0.0,
    seed: int = 42,
) -> SplitterDataModule:
    """
    Load data and build a SplitterDataModule in one step.
    
    This avoids materializing a large list of dictionaries between steps.
    
    Args:
        file_pattern: Glob pattern for data files (required, but can be passed via parameters)
        num_workers: Number of DataLoader workers. If None, auto-detects based on CPU count.
            Set to 0 to disable multiprocessing.
    """
    import os
    import sys
    
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
    groups = load_splitter_groups(
        file_pattern,
        max_files=max_files,
        limit_groups=limit_groups,
        min_hits=min_hits,
        verbose=True,
    )
    print(f"Loaded {len(groups)} groups. Building datamodule...", file=sys.stderr, flush=True)
    datamodule = SplitterDataModule(
        records=groups,
        use_group_probs=use_group_probs,
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
def run_splitter_hparam_search(
    datamodule: SplitterDataModule,
    n_trials: int = 25,
    max_epochs: int = 20,
    limit_train_batches: int | float | None = 0.8,
    limit_val_batches: int | float | None = 1.0,
) -> Dict[str, Any]:
    """
    Run Optuna hyperparameter search for group splitter.
    """
    import sys
    import torch
    
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

    def objective(trial: optuna.Trial) -> float:
        trial_num = trial.number + 1
        print(f"Trial {trial_num}/{n_trials} starting...", file=sys.stderr, flush=True)
        
        batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
        hidden = trial.suggest_categorical("hidden", [128, 150, 192])
        heads = trial.suggest_int("heads", 6, 12)
        layers = trial.suggest_int("layers", 2, 4)
        dropout = trial.suggest_float("dropout", 0.05, 0.25)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

        print(f"Trial {trial_num} params: batch_size={batch_size}, hidden={hidden}, heads={heads}, layers={layers}, dropout={dropout:.4f}, lr={lr:.6f}, weight_decay={weight_decay:.6f}", file=sys.stderr, flush=True)

        datamodule.batch_size = batch_size

        # Determine input channels based on use_group_probs
        in_channels = 8 if datamodule.use_group_probs else 5
        model = GroupSplitter(
            in_channels=in_channels,
            hidden=hidden,
            heads=heads,
            layers=layers,
            dropout=dropout,
            num_classes=NUM_NODE_CLASSES,
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

    study = optuna.create_study(direction="maximize")
    print(f"Starting Optuna study...", file=sys.stderr, flush=True)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"Optuna search complete! Best score: {study.best_value:.6f}", file=sys.stderr, flush=True)
    print(f"Best params: {study.best_params}", file=sys.stderr, flush=True)

    best_params = study.best_params
    best_params["best_score"] = study.best_value
    best_params["n_trials"] = len(study.trials)
    return best_params


@step(enable_cache=False)
def train_best_splitter(
    best_params: Dict[str, Any],
    datamodule: SplitterDataModule,
    max_epochs: int = 50,
    early_stopping: bool = True,
    early_stopping_patience: int = 6,
    early_stopping_monitor: str = "val_loss",
    early_stopping_mode: str = "min",
) -> GraphLightningModule:
    """Train the final group splitter using the best hyperparameters."""
    datamodule.setup(stage="fit")
    if "batch_size" in best_params:
        datamodule.batch_size = int(best_params["batch_size"])

    in_channels = 8 if datamodule.use_group_probs else 5
    model = GroupSplitter(
        in_channels=in_channels,
        hidden=int(best_params.get("hidden", 150)),
        heads=int(best_params.get("heads", 10)),
        layers=int(best_params.get("layers", 3)),
        dropout=float(best_params.get("dropout", 0.15)),
        num_classes=NUM_NODE_CLASSES,
    )
    lightning_module = GraphLightningModule(
        model,
        task="classification",
        lr=float(best_params.get("lr", 1e-3)),
        weight_decay=float(best_params.get("weight_decay", 1e-4)),
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

    min_epochs = max_epochs if not early_stopping else None

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        gradient_clip_val=2.0,
        callbacks=callbacks,
    )

    def fit():
        trainer.fit(lightning_module, datamodule=datamodule)

    _run_silently(fit)
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


@step(
    enable_cache=False,
    output_materializers=(TorchTensorMaterializer, TorchTensorMaterializer),
)
def collect_splitter_predictions(
    module: GraphLightningModule, datamodule: SplitterDataModule
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
            out = module(batch).detach().cpu()  # [total_nodes, 3]
        target = batch.y.detach().cpu()  # [total_nodes, 3]

        preds.append(out)
        targets.append(target)

    return torch.cat(preds), torch.cat(targets)


@pipeline
def group_splitter_optuna_pipeline(
    build_datamodule_params: Optional[Mapping[str, Any]] = None,
    run_hparam_search_params: Optional[Mapping[str, Any]] = None,
    train_best_model_params: Optional[Mapping[str, Any]] = None,
):
    """
    Complete pipeline for training group splitter with Optuna hyperparameter tuning.
    
    Args:
        build_datamodule_params: Parameters for build_splitter_datamodule
            (includes file_pattern and all data loading + datamodule params)
        run_hparam_search_params: Parameters for run_splitter_hparam_search
        train_best_model_params: Parameters for train_best_splitter
    """
    dm_kwargs = dict(build_datamodule_params or {})
    search_kwargs = dict(run_hparam_search_params or {})
    train_kwargs = dict(train_best_model_params or {})

    # Explicitly extract and pass each parameter (ZenML has issues with **kwargs unpacking)
    datamodule = build_splitter_datamodule(
        file_pattern=dm_kwargs.get("file_pattern"),
        max_files=dm_kwargs.get("max_files"),
        limit_groups=dm_kwargs.get("limit_groups"),
        min_hits=dm_kwargs.get("min_hits", 3),
        use_group_probs=dm_kwargs.get("use_group_probs", False),
        batch_size=dm_kwargs.get("batch_size", 8),
        num_workers=dm_kwargs.get("num_workers"),
        pin_memory=dm_kwargs.get("pin_memory", False),
        val_split=dm_kwargs.get("val_split", 0.15),
        test_split=dm_kwargs.get("test_split", 0.0),
        seed=dm_kwargs.get("seed", 42),
    )
    best_params = run_splitter_hparam_search(
        datamodule,
        n_trials=search_kwargs.get("n_trials", 25),
        max_epochs=search_kwargs.get("max_epochs", 20),
        limit_train_batches=search_kwargs.get("limit_train_batches", 0.8),
        limit_val_batches=search_kwargs.get("limit_val_batches", 1.0),
    )
    trained_module = train_best_splitter(
        best_params,
        datamodule,
        max_epochs=train_kwargs.get("max_epochs", 50),
        early_stopping=train_kwargs.get("early_stopping", True),
        early_stopping_patience=train_kwargs.get("early_stopping_patience", 6),
        early_stopping_monitor=train_kwargs.get("early_stopping_monitor", "val_loss"),
        early_stopping_mode=train_kwargs.get("early_stopping_mode", "min"),
    )
    predictions, targets = collect_splitter_predictions(trained_module, datamodule)
    return trained_module, datamodule, predictions, targets, best_params

