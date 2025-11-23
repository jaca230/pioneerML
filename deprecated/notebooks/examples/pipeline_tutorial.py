"""
Runnable version of the pipeline tutorial notebook.

Generates a synthetic dataset with random characteristics to exercise the
pipeline + Lightning integration. Each event has:
- Random number of hits (5-40)
- Random spatial/energy features
- Multi-label targets based on physics-inspired criteria

Trains for 5 epochs on all data (no batch limits) to demonstrate the full workflow.
Includes comprehensive evaluation with metrics, confusion matrices, and example predictions.

Defaults to CPU to avoid GPU capability issues; override via CLI if desired.
Use --n-events to control dataset size (default: 10,000 events).
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
    CleanProgressBar,
)

# Additional imports for evaluation
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402
import seaborn as sns  # noqa: E402


def make_record(num_hits: int, event_id: int) -> dict:
    """Generate a synthetic detector event with random hit data."""
    coord = np.random.randn(num_hits).astype(np.float32)
    z = np.random.randn(num_hits).astype(np.float32)
    energy = np.abs(np.random.randn(num_hits)).astype(np.float32)
    view = np.random.randint(0, 2, num_hits).astype(np.float32)

    # Multi-label targets (3 classes) based on physics-inspired features
    # Class 0: High energy events (energy mean > threshold)
    # Class 1: Events with many hits (num_hits > threshold)
    # Class 2: Localized events (spatial spread < threshold)

    energy_threshold = 0.8 + np.random.randn() * 0.2  # Random threshold
    hit_threshold = 15 + np.random.randint(-5, 5)  # Random threshold
    spatial_spread = np.std(coord)
    energy_mean = energy.mean()

    labels = [
        int(energy_mean > energy_threshold),
        int(num_hits > hit_threshold),
        int(spatial_spread < 1.0),
    ]

    return {
        "coord": coord,
        "z": z,
        "energy": energy,
        "view": view,
        "labels": labels,
        "event_id": event_id,
        "group_id": event_id,
        # Store decision boundary features for visualization
        "energy_mean": energy_mean,
        "energy_threshold": energy_threshold,
        "num_hits": num_hits,
        "hit_threshold": hit_threshold,
        "spatial_spread": spatial_spread,
    }


def build_dataset(n: int = 10_000, seed: int = 42) -> tuple[GraphGroupDataset, list[dict]]:
    """Build a synthetic dataset with n samples and random characteristics.

    Args:
        n: Number of events to generate
        seed: Random seed for reproducibility

    Returns:
        Tuple of (GraphGroupDataset, list of record dicts with metadata)
    """
    np.random.seed(seed)
    records: list[dict] = [
        make_record(
            num_hits=np.random.randint(5, 40),  # Random hit count between 5-40
            event_id=i,
        )
        for i in range(n)
    ]
    return GraphGroupDataset(records, num_classes=3), records


def main(device: str, max_epochs: int, num_workers: int, plot_path: str | None, n_events: int = 10_000) -> None:
    print(f"Building synthetic dataset with {n_events:,} events...")
    dataset, records = build_dataset(n=n_events)
    print(f"  Created {len(dataset):,} samples with random hit counts (5-40 per event)")
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
        # Process all batches each epoch (no limits)
        "logger": False,
        "enable_checkpointing": False,
        "precision": default_precision_for_accelerator(device),
        "enable_model_summary": False,
        "enable_progress_bar": False,  # Disable default progress bar
        "callbacks": [CleanProgressBar(bar_width=20)],  # Use custom progress bar
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

    # Per-epoch loss curve (aligned lengths - both train and val run each epoch)
    if lightning_module.train_epoch_loss_history:
        plot_loss_curves(
            train_losses=lightning_module.train_epoch_loss_history,
            val_losses=lightning_module.val_epoch_loss_history,
            title="Loss per epoch",
            xlabel="Epoch",
            save_path=plot_path,
            show=plot_path is None,
        )
        if plot_path:
            print(f"Saved loss plot to {plot_path}")
        print(f"\nFinal training loss: {lightning_module.train_epoch_loss_history[-1]:.4f}")
        print(f"Final validation loss: {lightning_module.val_epoch_loss_history[-1]:.4f}")

    # Evaluation: Get predictions on validation set
    print("\n" + "=" * 80)
    print("EVALUATING MODEL ON VALIDATION SET")
    print("=" * 80)

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    num_classes = 3
    with torch.no_grad():
        for batch in datamodule.val_dataloader():
            batch_size = batch.num_graphs  # Number of graphs in this batch
            logits = model(batch)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_preds.append(preds.cpu().numpy())
            # batch.y is concatenated by PyG for graph-level tasks, reshape it
            labels = batch.y.cpu().numpy().reshape(batch_size, num_classes)
            all_labels.append(labels)
            all_probs.append(probs.cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    # Calculate metrics
    exact_match = (all_preds == all_labels).all(axis=1).mean()
    per_label_accuracy = (all_preds == all_labels).mean()

    print(f"\nValidation set size: {len(all_labels)} samples")
    print(f"\nOverall Performance:")
    print(f"  Exact Match Accuracy (all 3 labels correct): {exact_match:.2%}")
    print(f"  Per-Label Average Accuracy: {per_label_accuracy:.2%}")

    # Per-class accuracy
    print(f"\nPer-class accuracies:")
    class_names = ["High Energy", "Many Hits", "Localized"]
    for class_idx in range(3):
        accuracy = (all_preds[:, class_idx] == all_labels[:, class_idx]).mean()
        print(f"  {class_names[class_idx]}: {accuracy:.2%}")

    # Show example predictions
    print(f"\nExample predictions (first 10):")
    print(f"{'Sample':<8} {'True':<15} {'Predicted':<15} {'Probabilities':<25} {'Match'}")
    print("-" * 75)
    num_examples = min(10, len(all_labels))
    for i in range(num_examples):
        true_labels = all_labels[i].astype(int)
        pred_labels = all_preds[i].astype(int)
        probs = all_probs[i]

        true_str = str(list(true_labels))
        pred_str = str(list(pred_labels))
        prob_str = f"[{probs[0]:.2f}, {probs[1]:.2f}, {probs[2]:.2f}]"
        match = "✓" if np.array_equal(true_labels, pred_labels) else "✗"

        print(f"{i:<8} {true_str:<15} {pred_str:<15} {prob_str:<25} {match}")

    # Visualize decision boundaries and model predictions
    print("\n" + "=" * 80)
    print("DECISION BOUNDARY VISUALIZATION")
    print("=" * 80)

    # Get validation indices (need to figure out which records are in validation set)
    datamodule.setup()
    val_indices = datamodule.val_dataset.indices

    # Extract metadata for validation samples
    val_records = [records[i] for i in val_indices]

    # Extract features and thresholds
    energy_means = np.array([r["energy_mean"] for r in val_records])
    energy_thresholds = np.array([r["energy_threshold"] for r in val_records])
    num_hits = np.array([r["num_hits"] for r in val_records])
    hit_thresholds = np.array([r["hit_threshold"] for r in val_records])
    spatial_spreads = np.array([r["spatial_spread"] for r in val_records])

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Class 0: Energy threshold visualization
    ax = axes[0]
    correct_mask = all_preds[:, 0] == all_labels[:, 0]

    # Plot points colored by prediction - use explicit colors for binary classes
    colors = ['#d62728' if pred == 0 else '#2ca02c' for pred in all_preds[:, 0]]
    scatter = ax.scatter(
        range(len(energy_means)),
        energy_means,
        c=colors,
        alpha=0.6,
        edgecolors='black',
        linewidths=0.5,
        s=50,
        label='Predictions'
    )

    # Plot the individual thresholds as a line
    ax.plot(range(len(energy_thresholds)), energy_thresholds,
            'b-', alpha=0.3, linewidth=1, label='Truth Threshold')

    # Mark incorrect predictions with X
    incorrect_indices = np.where(~correct_mask)[0]
    if len(incorrect_indices) > 0:
        ax.scatter(incorrect_indices, energy_means[incorrect_indices],
                   marker='x', s=100, c='darkred', linewidths=2, label='Incorrect', zorder=5)

    # Add legend entries for colors
    legend_elements = [
        Patch(facecolor='#d62728', label='Predicted: Low Energy (0)'),
        Patch(facecolor='#2ca02c', label='Predicted: High Energy (1)'),
        ax.plot([], [], 'b-', alpha=0.3, linewidth=1)[0],
    ]
    if len(incorrect_indices) > 0:
        legend_elements.append(ax.scatter([], [], marker='x', s=100, c='darkred', linewidths=2))

    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Mean Energy')
    ax.set_title('Class 0: High Energy Classification')
    ax.legend(handles=legend_elements, labels=['Predicted: Low (0)', 'Predicted: High (1)', 'Truth Threshold', 'Incorrect'] if len(incorrect_indices) > 0 else ['Predicted: Low (0)', 'Predicted: High (1)', 'Truth Threshold'])
    ax.grid(True, alpha=0.3)

    # Class 1: Hit count visualization
    ax = axes[1]
    correct_mask = all_preds[:, 1] == all_labels[:, 1]

    # Plot points colored by prediction
    colors = ['#d62728' if pred == 0 else '#2ca02c' for pred in all_preds[:, 1]]
    scatter = ax.scatter(
        range(len(num_hits)),
        num_hits,
        c=colors,
        alpha=0.6,
        edgecolors='black',
        linewidths=0.5,
        s=50
    )

    # Plot the individual thresholds
    ax.plot(range(len(hit_thresholds)), hit_thresholds,
            'b-', alpha=0.3, linewidth=1, label='Truth Threshold')

    # Mark incorrect predictions
    incorrect_indices = np.where(~correct_mask)[0]
    if len(incorrect_indices) > 0:
        ax.scatter(incorrect_indices, num_hits[incorrect_indices],
                   marker='x', s=100, c='darkred', linewidths=2, label='Incorrect', zorder=5)

    # Add legend
    legend_elements = [
        Patch(facecolor='#d62728', label='Predicted: Few Hits (0)'),
        Patch(facecolor='#2ca02c', label='Predicted: Many Hits (1)'),
        ax.plot([], [], 'b-', alpha=0.3, linewidth=1)[0],
    ]
    if len(incorrect_indices) > 0:
        legend_elements.append(ax.scatter([], [], marker='x', s=100, c='darkred', linewidths=2))

    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Number of Hits')
    ax.set_title('Class 1: Many Hits Classification')
    ax.legend(handles=legend_elements, labels=['Predicted: Few (0)', 'Predicted: Many (1)', 'Truth Threshold', 'Incorrect'] if len(incorrect_indices) > 0 else ['Predicted: Few (0)', 'Predicted: Many (1)', 'Truth Threshold'])
    ax.grid(True, alpha=0.3)

    # Class 2: Spatial spread visualization
    ax = axes[2]
    correct_mask = all_preds[:, 2] == all_labels[:, 2]

    # Plot points colored by prediction
    colors = ['#d62728' if pred == 0 else '#2ca02c' for pred in all_preds[:, 2]]
    scatter = ax.scatter(
        range(len(spatial_spreads)),
        spatial_spreads,
        c=colors,
        alpha=0.6,
        edgecolors='black',
        linewidths=0.5,
        s=50
    )

    # Plot the threshold (fixed at 1.0)
    ax.axhline(y=1.0, color='b', linestyle='-', alpha=0.3, linewidth=1, label='Truth Threshold (1.0)')

    # Mark incorrect predictions
    incorrect_indices = np.where(~correct_mask)[0]
    if len(incorrect_indices) > 0:
        ax.scatter(incorrect_indices, spatial_spreads[incorrect_indices],
                   marker='x', s=100, c='darkred', linewidths=2, label='Incorrect', zorder=5)

    # Add legend
    legend_elements = [
        Patch(facecolor='#d62728', label='Predicted: Spread Out (0)'),
        Patch(facecolor='#2ca02c', label='Predicted: Localized (1)'),
        ax.axhline(y=0, color='b', linestyle='-', alpha=0.3, linewidth=1),
    ]
    if len(incorrect_indices) > 0:
        legend_elements.append(ax.scatter([], [], marker='x', s=100, c='darkred', linewidths=2))

    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Spatial Spread (std)')
    ax.set_title('Class 2: Localized Event Classification')
    ax.legend(handles=legend_elements, labels=['Predicted: Spread (0)', 'Predicted: Localized (1)', 'Truth Threshold (1.0)', 'Incorrect'] if len(incorrect_indices) > 0 else ['Predicted: Spread (0)', 'Predicted: Localized (1)', 'Truth Threshold (1.0)'])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\nInterpretation:")
    print("- Green points: Model predicted class 1 (above threshold)")
    print("- Red points: Model predicted class 0 (below threshold)")
    print("- Blue line: Ground truth decision boundary")
    print("- Red X markers: Incorrect predictions")
    print("\nGood performance = points are green above threshold and red below threshold")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "gpu", "mps", "auto"],
        help="Accelerator to use (default: cpu).",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs (default: 5).")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--n-events", type=int, default=10_000, help="Number of synthetic events to generate (default: 10,000).")
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
        num_workers=args.num_workers,
        plot_path=args.plot_path,
        n_events=args.n_events,
    )
