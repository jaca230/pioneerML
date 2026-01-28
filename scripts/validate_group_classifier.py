"""
Run group-classifier validation (non-notebook) and generate plots.

Usage:
    python scripts/validate_group_classifier.py [--checkpoint PATH] [--max-files 2]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch_geometric.loader import DataLoader

from pioneerml.data import CLASS_NAMES, NUM_GROUP_CLASSES
from pioneerml.evaluation.plots import (
    plot_confidence_analysis,
    plot_embedding_space,
    plot_multilabel_confusion_matrix,
    plot_precision_recall_curves,
    plot_probability_distributions,
    plot_roc_curves,
)
from pioneerml.models.classifiers.group_classifier import GroupClassifier
from pioneerml.training.datamodules import GroupClassificationDataModule
from pioneerml.zenml import utils as zenml_utils
from pioneerml.zenml.pipelines.training.group_classification import GroupClassificationProcessor
from pioneerml.zenml.pipelines.training.group_classification.loader import GroupClassificationLoader


def _select_checkpoint(checkpoints_dir: Path, explicit: Optional[Path]) -> tuple[Path, Optional[dict]]:
    if explicit is not None:
        ckpt = explicit
        meta_path = ckpt.with_name(f"{ckpt.stem}_metadata.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
        return ckpt, meta

    checkpoint_files = sorted(checkpoints_dir.glob("group_classifier_*.pt"), reverse=True)
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")

    ckpt = checkpoint_files[0]
    timestamp = ckpt.stem.replace("group_classifier_", "")
    meta_path = checkpoints_dir / f"group_classifier_{timestamp}_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    return ckpt, meta


def _build_model(metadata: Optional[dict]) -> GroupClassifier:
    arch = metadata.get("model_architecture", {}) if metadata else {}
    best_params = metadata.get("best_hyperparameters", {}) if metadata else {}

    hidden = arch.get("hidden") or best_params.get("hidden", 192)
    num_blocks = arch.get("num_blocks") or best_params.get("num_blocks", 3)
    dropout = arch.get("dropout") or best_params.get("dropout", 0.1)
    num_classes = arch.get("num_classes") or best_params.get("num_classes", NUM_GROUP_CLASSES)

    return GroupClassifier(
        hidden=int(hidden),
        num_blocks=int(num_blocks),
        dropout=float(dropout),
        num_classes=int(num_classes),
    )


def _load_data(
    project_root: Path,
    num_classes: int,
    *,
    parquet_pattern: str,
    max_files: Optional[int],
    limit_groups: Optional[int],
    max_hits: int,
    batch_size: int,
    num_workers: int,
    val_split: float,
) -> tuple[GroupClassificationDataModule, Sequence[str]]:
    processor = GroupClassificationProcessor(max_hits=max_hits)
    loader = GroupClassificationLoader(columns=processor.columns)
    df = loader.load(parquet_pattern, max_files=max_files, limit_groups=limit_groups)
    groups = processor.process(df)

    dm = GroupClassificationDataModule(
        records=groups,
        num_classes=num_classes,
        batch_size=batch_size,
        num_workers=num_workers,
        val_split=val_split,
        test_split=0.0,
        seed=42,
    )
    dm.setup(stage="fit")

    val_dataset = dm.val_dataset or dm.train_dataset
    if val_dataset is None:
        raise ValueError("No validation dataset available")

    return dm, list(CLASS_NAMES.values())


def run(args: argparse.Namespace) -> None:
    project_root = zenml_utils.find_project_root()
    checkpoints_dir = Path(project_root) / "trained_models" / "group_classifier"
    ckpt_path, metadata = _select_checkpoint(checkpoints_dir, args.checkpoint)

    model = _build_model(metadata)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    datamodule, default_class_names = _load_data(
        project_root,
        num_classes=model.num_classes,
        parquet_pattern=args.parquet_pattern or str(Path(project_root) / "data" / "ml_output_*.parquet"),
        max_files=args.max_files,
        limit_groups=args.limit_groups,
        max_hits=args.max_hits,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
    )

    val_dataset = datamodule.val_dataset or datamodule.train_dataset
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            all_predictions.append(model(batch).cpu())
            all_targets.append(batch.y.cpu())

    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)

    print(f"Predictions: {predictions.shape}  Targets: {targets.shape}")

    plots_dir = Path(project_root) / "artifacts" / "validation_plots" / "group_classifier"
    plots_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = (metadata or {}).get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
    plot_prefix = f"group_classifier_{timestamp_str}"
    class_names = (metadata or {}).get("dataset_info", {}).get("class_names", default_class_names)

    confusion_path = plots_dir / f"{plot_prefix}_confusion_matrix.png"
    roc_path = plots_dir / f"{plot_prefix}_roc_curves.png"
    pr_path = plots_dir / f"{plot_prefix}_precision_recall.png"
    embed_path = plots_dir / f"{plot_prefix}_embedding_space_tsne.png"

    print("Generating confusion matrix...")
    plot_multilabel_confusion_matrix(
        predictions=predictions,
        targets=targets,
        class_names=class_names,
        threshold=0.5,
        normalize=True,
        save_path=confusion_path,
        show=False,
    )
    print(f"Saved: {confusion_path}")

    print("Generating ROC curves...")
    plot_roc_curves(
        predictions=predictions,
        targets=targets,
        class_names=class_names,
        save_path=roc_path,
        show=False,
    )
    print(f"Saved: {roc_path}")

    print("Generating precision-recall curves...")
    plot_precision_recall_curves(
        predictions=predictions,
        targets=targets,
        class_names=class_names,
        save_path=pr_path,
        show=False,
    )
    print(f"Saved: {pr_path}")

    # Optional heavier plots
    if args.embeddings:
        print("Generating embedding space visualization (t-SNE)...")
        embeddings_list = []
        targets_list = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                embeddings_list.append(model.extract_embeddings(batch).cpu())
                targets_list.append(batch.y.cpu())
        embeddings = torch.cat(embeddings_list, dim=0)
        targets_for_emb = torch.cat(targets_list, dim=0)
        plot_embedding_space(
            embeddings=embeddings,
            targets=targets_for_emb,
            class_names=class_names,
            method="tsne",
            perplexity=30.0,
            n_components=2,
            save_path=embed_path,
            show=False,
        )
        print(f"Saved: {embed_path}")

    if args.probabilities:
        print("Generating probability distribution plots...")
        plot_probability_distributions(
            predictions=predictions,
            targets=targets,
            class_names=class_names,
            show=False,
        )
        print("Probability distribution plots generated (not saved by default).")

    if args.confidence:
        print("Generating confidence analysis plots...")
        plot_confidence_analysis(
            predictions=predictions,
            targets=targets,
            class_names=class_names,
            show=False,
        )
        print("Confidence analysis plots generated (not saved by default).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate group classifier checkpoints.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to a .pt checkpoint to evaluate.")
    parser.add_argument(
        "--parquet-pattern",
        type=str,
        default=None,
        help="Glob for ml_output parquet shards (default: data/ml_output_*.parquet).",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Max input files to load for validation.")
    parser.add_argument("--limit-groups", type=int, default=None, help="Limit total groups for faster runs.")
    parser.add_argument("--max-hits", type=int, default=256, help="Pad/truncate hits to this length.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for validation loader.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers.")
    parser.add_argument("--val-split", type=float, default=0.0, help="Validation split ratio (0 uses full set).")
    parser.add_argument("--embeddings", action="store_true", help="Generate embedding t-SNE plot.")
    parser.add_argument("--probabilities", action="store_true", help="Generate probability distribution plots.")
    parser.add_argument("--confidence", action="store_true", help="Generate confidence analysis plots.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
