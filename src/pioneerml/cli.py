"""
Command-line interface for PIONEER ML.

This will provide commands for:
- Training models
- Running inference
- Evaluating checkpoints
- Pipeline execution
- Data preprocessing
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from pioneerml.evaluation import MetricCollection, PLOT_REGISTRY, default_metrics_for_task


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="pioneerml",
        description="PIONEER ML: Machine Learning Pipeline Framework",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command (placeholder)
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config", type=str, help="Path to config file")
    train_parser.add_argument("--model", type=str, help="Model name")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a model")
    eval_parser.add_argument("--predictions", type=str, required=True, help="Path to predictions (.npy or .pt)")
    eval_parser.add_argument("--targets", type=str, required=True, help="Path to targets (.npy or .pt)")
    eval_parser.add_argument(
        "--task",
        type=str,
        default="multilabel",
        choices=["multilabel", "regression", "classification", "multi-label"],
        help="Task type for default metrics",
    )
    eval_parser.add_argument(
        "--metrics",
        type=str,
        nargs="*",
        help="Metric names to compute (defaults depend on task)",
    )
    eval_parser.add_argument(
        "--plots",
        type=str,
        nargs="*",
        default=[],
        help=f"Plot names to generate (available: {list(PLOT_REGISTRY)})",
    )
    eval_parser.add_argument(
        "--class-names",
        type=str,
        nargs="*",
        help="Optional class names for labeling metrics/plots",
    )
    eval_parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for multilabel classification",
    )
    eval_parser.add_argument(
        "--save-dir",
        type=str,
        default="outputs/evaluation",
        help="Directory to store generated plots",
    )
    eval_parser.add_argument(
        "--save-json",
        type=str,
        help="Optional path to write metrics JSON",
    )

    # Predict command (placeholder)
    predict_parser = subparsers.add_parser("predict", help="Run inference")
    predict_parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    predict_parser.add_argument("--input", type=str, help="Path to input data")
    predict_parser.add_argument("--output", type=str, help="Path to save predictions")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Placeholder implementations
    if args.command == "train":
        print("Training functionality coming soon!")
        print("For now, use the Jupyter notebooks in the repository.")
        return 1
    elif args.command == "evaluate":
        return _run_evaluate(args)
    elif args.command == "predict":
        print("Prediction functionality coming soon!")
        return 1

    return 0


def _load_array(path: str):
    path_obj = Path(path)
    if path_obj.suffix in {".pt", ".pth"}:
        return torch.load(path_obj)
    return np.load(path_obj)


def _run_evaluate(args) -> int:
    predictions = _load_array(args.predictions)
    targets = _load_array(args.targets)

    metric_names = args.metrics or default_metrics_for_task(args.task)
    metric_params = {"threshold": args.threshold}
    if args.class_names:
        metric_params["class_names"] = args.class_names

    collection = MetricCollection.from_names(metric_names)
    metrics = collection(predictions, targets, **metric_params)

    generated_plots = {}
    if args.plots:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for plot_name in args.plots:
            plot_fn = PLOT_REGISTRY.get(plot_name)
            if plot_fn is None:
                print(f"[warn] Plot '{plot_name}' is not registered. Available: {list(PLOT_REGISTRY)}")
                continue
            plot_path = save_dir / f"{plot_name}.png"
            generated_plots[plot_name] = plot_fn(
                predictions=predictions,
                targets=targets,
                class_names=args.class_names,
                save_path=plot_path,
            )
    if generated_plots:
        metrics["plots"] = generated_plots

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
