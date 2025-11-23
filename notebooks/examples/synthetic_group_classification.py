"""
Synthetic group classification training demo using the PIONEER ML pipeline.

What it does:
- Detects GPU/MPS/CPU and configures Lightning accordingly.
- Generates synthetic graph records with three binary labels:
  * High vs low total energy
  * High vs low hit count
  * Wide vs narrow spatial spread (std > 1)
- Wraps the synthetic data in the GroupClassificationDataModule.
- Trains a GroupClassifier through the Pipeline + LightningTrainStage wrapper.
- Plots train/val loss and per-class predictions vs the underlying metric.

Usage:
    python notebooks/examples/synthetic_group_classification.py

Outputs are written to notebooks/examples/output/.
"""

from __future__ import annotations

import sys
from pathlib import Path

# --- Auto-detect project root by walking upward ---
cwd = Path().resolve()
ROOT = None

for parent in [cwd] + list(cwd.parents):
    if (parent / "src" / "pioneerml").exists():
        ROOT = parent
        break

if ROOT is None:
    raise RuntimeError("Could not find project root containing src/pioneerml")

sys.path.append(str(ROOT / "src"))
print("Using project root:", ROOT)


import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import pytorch_lightning as pl
import seaborn as sns
from sklearn.metrics import confusion_matrix

from pioneerml.data.datasets.graph_group import GraphRecord
from pioneerml.training.datamodules import GroupClassificationDataModule
from pioneerml.models import GroupClassifier
from pioneerml.training import (
    GraphLightningModule,
    CleanProgressBar,
    default_precision_for_accelerator,
)
from pioneerml.pipelines import Pipeline, Context
from pioneerml.pipelines.stage import StageConfig
from pioneerml.pipelines.stages import LightningTrainStage


# Thresholds for synthetic labels
ENERGY_THRESHOLD = 25.0
HIT_THRESHOLD = 18
SPREAD_THRESHOLD = 1.0  # std > 1 counts as wide

CLASS_CONFIG = [
    ("High energy", "total_energy", ENERGY_THRESHOLD),
    ("High hit count", "num_hits", HIT_THRESHOLD),
    ("Wide spatial spread", "spread", SPREAD_THRESHOLD),
]


def choose_device() -> Tuple[str, int, torch.device]:
    """Pick accelerator/devices/device triple for Lightning and torch."""
    if torch.cuda.is_available():
        return "gpu", 1, torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", 1, torch.device("mps")
    return "cpu", 1, torch.device("cpu")


def generate_synthetic_record(idx: int, rng: np.random.Generator) -> GraphRecord:
    """Create one synthetic GraphRecord with metric-driven labels."""
    num_hits = int(rng.integers(6, 40))
    spread_scale = float(rng.choice([0.35, 0.6, 1.4]))

    coord = rng.normal(loc=rng.normal(0.0, 0.5), scale=spread_scale, size=num_hits)
    z = rng.normal(
        loc=rng.normal(0.2, 0.6), scale=spread_scale * rng.uniform(0.8, 1.3), size=num_hits
    )
    energy = rng.gamma(shape=rng.uniform(1.2, 2.0), scale=rng.uniform(2.5, 4.5), size=num_hits)
    view = rng.integers(0, 2, size=num_hits)

    total_energy = float(energy.sum())
    spread = float(np.sqrt(coord.std() ** 2 + z.std() ** 2))

    high_energy = total_energy > ENERGY_THRESHOLD
    high_hits = num_hits > HIT_THRESHOLD
    wide_spread = spread > SPREAD_THRESHOLD

    labels: list[int] = []
    if high_energy:
        labels.append(0)
    if high_hits:
        labels.append(1)
    if wide_spread:
        labels.append(2)

    record = GraphRecord(
        coord=coord,
        z=z,
        energy=energy,
        view=view,
        labels=labels,
        event_id=idx,
        group_id=idx,
    )
    # Keep metrics for visualization later.
    record.metrics = {
        "total_energy": total_energy,
        "num_hits": num_hits,
        "spread": spread,
        "high_energy": high_energy,
        "high_hit_count": high_hits,
        "wide_spread": wide_spread,
    }
    return record


def build_dataset(num_samples: int, seed: int = 7) -> list[GraphRecord]:
    rng = np.random.default_rng(seed)
    return [generate_synthetic_record(i, rng) for i in range(num_samples)]


def build_datamodule(records: list[GraphRecord], batch_size: int, pin_memory: bool) -> GroupClassificationDataModule:
    dm = GroupClassificationDataModule(
        records,
        num_classes=3,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=pin_memory,
        val_split=0.2,
        seed=123,
    )
    dm.setup()
    return dm


def train_with_pipeline(
    datamodule: GroupClassificationDataModule,
    accelerator: str,
    devices: int,
    max_epochs: int,
) -> tuple[GraphLightningModule, pl.Trainer]:
    model = GroupClassifier(num_classes=3, hidden=128, num_blocks=2, heads=4, dropout=0.1)
    lightning_module = GraphLightningModule(
        model,
        task="classification",
        lr=1e-3,
        weight_decay=1e-4,
    )

    train_stage = LightningTrainStage(
        config=StageConfig(
            name="train_synthetic_classifier",
            params={
                "module": lightning_module,
                "datamodule": datamodule,
                "trainer_params": {
                    "accelerator": accelerator,
                    "devices": devices,
                    "max_epochs": max_epochs,
                    "logger": False,
                    "enable_checkpointing": False,
                    "precision": default_precision_for_accelerator(accelerator),
                    "enable_model_summary": True,
                    "enable_progress_bar": False,
                    "callbacks": [CleanProgressBar(bar_width=30)],
                },
            },
        )
    )

    pipeline = Pipeline(stages=[train_stage], name="synthetic_classification_pipeline")
    ctx = pipeline.run(Context())

    return ctx["lightning_module"], ctx["trainer"]


def plot_loss_curves(module: GraphLightningModule, out_dir: Path) -> Path:
    train_loss = module.train_epoch_loss_history
    val_loss = module.val_epoch_loss_history

    # Drop the optional sanity-check validation pass Lightning runs before training.
    if len(val_loss) > len(train_loss):
        val_loss = val_loss[1:]

    plt.figure(figsize=(7, 4))
    train_epochs = np.arange(1, len(train_loss) + 1)
    plt.plot(train_epochs, train_loss, label="Train")
    if val_loss:
        val_epochs = np.arange(1, len(val_loss) + 1)
        plt.plot(val_epochs, val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs validation loss")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    plt.tight_layout()

    out_path = out_dir / "loss_curves.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def collect_validation_outputs(
    module: GraphLightningModule, datamodule: GroupClassificationDataModule
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    module.eval()
    device = next(module.parameters()).device

    all_probs: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    metrics: list[dict] = []

    def _metrics_from_data(g) -> dict:
        # Prefer stored raw metrics, otherwise compute from node features.
        if hasattr(g, "_raw") and getattr(g._raw, "metrics", None) is not None:
            return g._raw.metrics

        x = g.x
        coord = x[:, 0].cpu().numpy()
        z = x[:, 1].cpu().numpy()
        energy = x[:, 2].cpu().numpy()
        total_energy = float(energy.sum())
        spread = float(np.sqrt(coord.std() ** 2 + z.std() ** 2))
        num_hits = int(x.shape[0])
        return {
            "total_energy": total_energy,
            "num_hits": num_hits,
            "spread": spread,
            "high_energy": total_energy > ENERGY_THRESHOLD,
            "high_hit_count": num_hits > HIT_THRESHOLD,
            "wide_spread": spread > SPREAD_THRESHOLD,
        }

    # Infer num_classes for reshaping safeguards
    num_classes = getattr(getattr(module, "model", module), "num_classes", None)
    if num_classes is None and hasattr(datamodule, "train_dataset") and datamodule.train_dataset is not None:
        base_ds = getattr(datamodule.train_dataset, "dataset", None)
        num_classes = getattr(base_ds, "num_classes", None)

    def _ensure_2d(arr: np.ndarray) -> np.ndarray:
        """Ensure predictions/targets are 2D [N, C]."""
        arr = np.asarray(arr)
        if arr.ndim == 1 and num_classes:
            if arr.size % num_classes == 0:
                return arr.reshape(-1, num_classes)
        if arr.ndim == 1:
            return arr[:, None]
        return arr

    with torch.no_grad():
        for batch in datamodule.val_dataloader():
            batch = batch.to(device)
            logits = module(batch)
            all_probs.append(torch.sigmoid(logits).cpu())
            all_targets.append(batch.y.cpu())
            metrics.extend([_metrics_from_data(g) for g in batch.to_data_list()])

    probs = torch.cat(all_probs, dim=0).numpy() if all_probs else np.zeros((0, 3))
    targets = torch.cat(all_targets, dim=0).numpy() if all_targets else np.zeros((0, 3))

    probs = _ensure_2d(probs)
    targets = _ensure_2d(targets)
    return probs, targets, metrics


def plot_class_predictions(
    probs: np.ndarray,
    targets: np.ndarray,
    metrics: list[dict],
    out_dir: Path,
) -> list[Path]:
    paths: list[Path] = []
    for class_idx, (label, metric_key, threshold) in enumerate(CLASS_CONFIG):
        if probs.size == 0 or targets.size == 0:
            print("Warning: empty validation predictions/targets; skipping class plots.")
            return paths
        values = np.array([m[metric_key] for m in metrics], dtype=float)
        preds_binary = (probs[:, class_idx] > 0.5).astype(int)
        truth = targets[:, class_idx].astype(int)
        incorrect = preds_binary != truth
        colors = np.where(preds_binary == 1, "tab:green", "tab:red")
        xs = np.arange(len(values))

        plt.figure(figsize=(9, 3))
        plt.scatter(xs, values, c=colors, alpha=0.7, label="Prediction")
        if incorrect.any():
            plt.scatter(
                xs[incorrect],
                values[incorrect],
                marker="x",
                color="black",
                s=80,
                label="Incorrect",
            )
        plt.axhline(threshold, color="k", linestyle="--", label=f"Truth threshold = {threshold}")
        plt.xlabel("Validation sample index")
        plt.ylabel(label)
        plt.title(f"{label}: green=pred 1, red=pred 0")
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
        plt.tight_layout()

        out_path = out_dir / f"class_{class_idx}_{metric_key}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        paths.append(out_path)
    return paths


def plot_confusion_matrices(
    probs: np.ndarray,
    targets: np.ndarray,
    out_dir: Path,
) -> Path:
    preds_binary = (probs > 0.5).astype(int)
    class_names = [cfg[0] for cfg in CLASS_CONFIG]

    fig, axes = plt.subplots(1, len(class_names), figsize=(15, 4))
    for class_idx, ax in enumerate(axes):
        y_true = targets[:, class_idx].flatten()
        y_pred = preds_binary[:, class_idx].flatten()
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            cbar=False,
        )
        ax.set_title(f"{class_names[class_idx]} Confusion Matrix")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    out_path = out_dir / "confusion_matrices.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_probability_histograms(
    probs: np.ndarray,
    targets: np.ndarray,
    out_dir: Path,
) -> Path:
    class_names = [cfg[0] for cfg in CLASS_CONFIG]
    fig, axes = plt.subplots(1, len(class_names), figsize=(15, 4))

    for class_idx, ax in enumerate(axes):
        y_true = targets[:, class_idx].flatten()
        pos_probs = probs[y_true == 1, class_idx]
        neg_probs = probs[y_true == 0, class_idx]

        ax.hist(neg_probs, bins=20, alpha=0.5, label="True Negative", color="red")
        ax.hist(pos_probs, bins=20, alpha=0.5, label="True Positive", color="blue")
        ax.axvline(0.5, color="black", linestyle="--", linewidth=2, label="Threshold (0.5)")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Count")
        ax.set_title(f"{class_names[class_idx]} - Probability Distribution")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)

    plt.tight_layout()
    out_path = out_dir / "probability_histograms.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic group classification demo")
    parser.add_argument("--num-samples", type=int, default=240, help="Synthetic graph count")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=15, help="Max training epochs")
    args = parser.parse_args()

    pl.seed_everything(123, workers=True)
    torch.set_float32_matmul_precision("medium")

    accelerator, devices, device = choose_device()
    pin_memory = accelerator in {"gpu", "cuda"}
    print(f"Using accelerator={accelerator}, devices={devices}, torch device={device}")

    records = build_dataset(args.num_samples, seed=7)
    print(
        f"First sample -> total_energy={records[0].metrics['total_energy']:.2f}, "
        f"hits={records[0].metrics['num_hits']}, spread={records[0].metrics['spread']:.2f}"
    )

    datamodule = build_datamodule(records, batch_size=args.batch_size, pin_memory=pin_memory)
    print(f"Train graphs: {len(datamodule.train_dataset)}, val graphs: {len(datamodule.val_dataset)}")

    lightning_module, trainer = train_with_pipeline(
        datamodule, accelerator=accelerator, devices=devices, max_epochs=args.epochs
    )
    print("Training complete.")

    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)

    loss_path = plot_loss_curves(lightning_module, out_dir)
    print(f"Saved loss curves -> {loss_path}")

    probs, targets, metrics = collect_validation_outputs(lightning_module, datamodule)
    if len(metrics) == 0:
        print("No validation samples to visualize.")
        return
    plot_paths = plot_class_predictions(probs, targets, metrics, out_dir)
    for p in plot_paths:
        print(f"Saved class plot -> {p}")

    cm_path = plot_confusion_matrices(probs, targets, out_dir)
    print(f"Saved confusion matrices -> {cm_path}")

    hist_path = plot_probability_histograms(probs, targets, out_dir)
    print(f"Saved probability histograms -> {hist_path}")


if __name__ == "__main__":
    main()
