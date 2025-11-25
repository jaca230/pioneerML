"""
Evaluation examples pipeline for tutorials.

This pipeline demonstrates various evaluation techniques and custom plots.
"""

import torch
from torch_geometric.data import Data
from zenml import pipeline, step

from pioneerml.models import GroupClassifier
from pioneerml.training import GraphDataModule, GraphLightningModule
from pioneerml.zenml.utils import detect_available_accelerator


def create_evaluation_data(num_samples: int = 150) -> list[Data]:
    """Create synthetic data for evaluation examples."""
    data = []
    for _ in range(num_samples):
        num_nodes = torch.randint(4, 8, (1,)).item()
        x = torch.randn(num_nodes, 5)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        edge_attr = torch.randn(edge_index.shape[1], 4)

        label = torch.randint(0, 3, (1,)).item()
        y = torch.zeros(3)
        y[label] = 1.0

        data.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    return data


@step
def prepare_evaluation_data() -> list[Data]:
    """Step to prepare evaluation data."""
    return create_evaluation_data(200)


@step
def prepare_evaluation_datamodule(data: list[Data]) -> GraphDataModule:
    """Step to create DataModule for evaluation."""
    return GraphDataModule(dataset=data, val_split=0.3, batch_size=16)


@step
def train_evaluation_model(datamodule: GraphDataModule) -> GraphLightningModule:
    """Step to train model for evaluation examples."""
    model = GroupClassifier(num_classes=3, hidden=128, num_blocks=2)
    lightning_module = GraphLightningModule(model, task="classification", lr=1e-3)

    accelerator, devices = detect_available_accelerator()
    import pytorch_lightning as pl

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=4,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    trainer.fit(lightning_module, datamodule=datamodule)
    return lightning_module.eval()


@step
def collect_predictions(
    trained_module: GraphLightningModule,
    datamodule: GraphDataModule
) -> tuple[torch.Tensor, torch.Tensor]:
    """Step to collect predictions and targets for evaluation."""
    val_loader = datamodule.val_dataloader()
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for batch in val_loader:
            out = trained_module(batch)
            preds_list.append(out.cpu())
            targets_list.append(batch.y.cpu())

    predictions = torch.cat(preds_list)
    targets = torch.cat(targets_list)

    return predictions, targets


@step
def compute_custom_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> dict:
    """Step to compute custom evaluation metrics."""
    # Convert logits to probabilities
    probs = torch.softmax(predictions, dim=1)

    # Get predicted classes
    preds = torch.argmax(probs, dim=1)
    true_classes = torch.argmax(targets, dim=1)

    # Calculate accuracy
    accuracy = (preds == true_classes).float().mean().item()

    # Calculate per-class accuracy
    per_class_acc = {}
    for class_idx in range(targets.shape[1]):
        class_mask = true_classes == class_idx
        if class_mask.sum() > 0:
            class_acc = (preds[class_mask] == class_idx).float().mean().item()
            per_class_acc[f"class_{class_idx}"] = class_acc

    # Calculate confidence scores
    confidence = torch.max(probs, dim=1)[0].mean().item()

    return {
        "accuracy": accuracy,
        "per_class_accuracy": per_class_acc,
        "mean_confidence": confidence,
        "num_samples": len(predictions),
    }


@step
def generate_evaluation_plots(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> dict:
    """Step to generate evaluation plots."""
    try:
        from pathlib import Path

        from pioneerml.evaluation.plots import (
            plot_multilabel_confusion_matrix,
            plot_roc_curves,
            plot_precision_recall_curves
        )

        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate plots and get paths
        plot_paths = {}

        # Confusion matrix
        plot_paths["confusion_matrix"] = plot_multilabel_confusion_matrix(
            predictions=predictions,
            targets=targets,
            class_names=["pi", "mu", "e+"],
            threshold=0.5,
            normalize=True,
            save_path=output_dir / "tutorial_eval_confusion.png",
            show=False
        )

        # ROC curves
        plot_paths["roc_curves"] = plot_roc_curves(
            predictions=predictions,
            targets=targets,
            class_names=["pi", "mu", "e+"],
            save_path=output_dir / "tutorial_eval_roc.png",
            show=False
        )

        # Precision-Recall curves
        plot_paths["pr_curves"] = plot_precision_recall_curves(
            predictions=predictions,
            targets=targets,
            class_names=["pi", "mu", "e+"],
            save_path=output_dir / "tutorial_eval_pr.png",
            show=False
        )

        return plot_paths

    except Exception as e:
        return {"error": f"Failed to generate plots: {str(e)}"}


@pipeline
def evaluation_examples_pipeline():
    """Evaluation examples pipeline."""
    data = prepare_evaluation_data()
    datamodule = prepare_evaluation_datamodule(data)
    trained_module = train_evaluation_model(datamodule)
    predictions, targets = collect_predictions(trained_module, datamodule)
    metrics = compute_custom_metrics(predictions, targets)
    plots = generate_evaluation_plots(predictions, targets)

    return {
        "model": trained_module,
        "datamodule": datamodule,
        "metrics": metrics,
        "plots": plots,
        "predictions": predictions,
        "targets": targets,
    }
