"""
Runnable version of the pipeline tutorial notebook.

Uses a tiny synthetic dataset to exercise the pipeline + Lightning integration.
Defaults to CPU to avoid GPU capability issues; override via CLI if desired.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Sequence

import numpy as np
import torch


def _add_repo_to_path() -> pathlib.Path:
    """Ensure repo root and src/ are on sys.path."""
    start = pathlib.Path.cwd()
    repo_root = start
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").exists():
            repo_root = path
            break
    src_dir = repo_root / "src"
    for candidate in (repo_root, src_dir):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    return repo_root


repo_root = _add_repo_to_path()

from pioneerml.data import GraphGroupDataset  # noqa: E402
from pioneerml.models import GroupClassifier  # noqa: E402
from pioneerml.pipelines import Context, Pipeline, StageConfig  # noqa: E402
from pioneerml.pipelines.stages import LightningTrainStage  # noqa: E402
from pioneerml.training import (  # noqa: E402
    GraphDataModule,
    GraphLightningModule,
    plot_loss_curves,
    set_tensor_core_precision,
    default_precision_for_accelerator,
)


def make_record(num_hits: int, event_id: int) -> dict:
    coord = np.random.randn(num_hits).astype(np.float32)
    z = np.random.randn(num_hits).astype(np.float32)
    energy = np.abs(np.random.randn(num_hits)).astype(np.float32)
    view = np.random.randint(0, 2, num_hits).astype(np.float32)

    labels = [int(energy.mean() > 0.5), int(num_hits % 2 == 0)]
    if len(labels) < 3:
        labels.append(0)

    return {
        "coord": coord,
        "z": z,
        "energy": energy,
        "view": view,
        "labels": labels,
        "event_id": event_id,
        "group_id": event_id,
    }


def build_dataset(n: int = 20) -> GraphGroupDataset:
    records: Sequence[dict] = [make_record(8 + i, i) for i in range(n)]
    return GraphGroupDataset(records, num_classes=3)


def main(device: str, max_epochs: int, limit_train_batches: int, num_workers: int, plot_path: str | None) -> None:
    dataset = build_dataset()
    datamodule = GraphDataModule(
        dataset=dataset,
        batch_size=4,
        val_split=0.2,
        test_split=0.0,
        num_workers=num_workers,
        pin_memory=device == "gpu",
    )

    model = GroupClassifier(num_classes=3, hidden=64, num_blocks=2)
    lightning_module = GraphLightningModule(model, task="classification", lr=1e-3)

    trainer_params = {
        "accelerator": device,
        "devices": 1,
        "max_epochs": max_epochs,
        "limit_train_batches": limit_train_batches,
        "limit_val_batches": 1,
        "logger": False,
        "enable_checkpointing": False,
        "precision": default_precision_for_accelerator(device),
        "enable_model_summary": False,
    }

    train_stage = LightningTrainStage(
        config=StageConfig(
            name="train",
            params={
                "module": lightning_module,
                "datamodule": datamodule,
                "trainer_params": trainer_params,
            },
        )
    )

    pipeline = Pipeline([train_stage], name="tutorial_pipeline")
    ctx = pipeline.run(Context())
    print("Context summary:", ctx.summary())
    print("Metrics:", ctx.get("metrics", {}))

    # Per-step loss curve (train has more steps than val because it has more batches)
    plot_loss_curves(
        train_losses=lightning_module.train_loss_history,
        val_losses=lightning_module.val_loss_history,
        title="Loss per step",
        xlabel="Step",
        save_path=plot_path,
        show=plot_path is None,
    )

    # Per-epoch loss curve (aligned lengths)
    if lightning_module.train_epoch_loss_history:
        plot_loss_curves(
            train_losses=lightning_module.train_epoch_loss_history,
            val_losses=lightning_module.val_epoch_loss_history,
            title="Loss per epoch",
            xlabel="Epoch",
            save_path=None if plot_path is None else plot_path.replace(".png", "_epoch.png"),
            show=plot_path is None,
        )
        if plot_path:
            print(f"Saved loss plots to {plot_path} (steps) and {plot_path.replace('.png', '_epoch.png')} (epochs)")
    elif plot_path:
        print(f"Saved loss plot to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "gpu", "mps", "auto"],
        help="Accelerator to use (default: cpu).",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--limit-train-batches", type=int, default=2, help="Limit train batches for quick runs.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--plot-path", type=str, default=None, help="Optional path to save loss plot.")
    args = parser.parse_args()

    # If user requested CUDA, make sure it's actually usable; otherwise fall back to CPU.
    requested_device = args.device
    if requested_device in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU.")
            requested_device = "cpu"
        else:
            props = torch.cuda.get_device_properties(0)
            print(f"Using CUDA device: {props.name} (cc {props.major}.{props.minor})")
            set_tensor_core_precision("medium")
    main(
        requested_device,
        max_epochs=args.epochs,
        limit_train_batches=args.limit_train_batches,
        num_workers=args.num_workers,
        plot_path=args.plot_path,
    )
