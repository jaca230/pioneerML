from pathlib import Path

import torch
from torch.utils.data import DataLoader
from zenml import step

from pioneerml.common.evaluation.plots.loss import LossCurvesPlot
from pioneerml.pipelines.training.group_classification_event.dataset import GroupClassifierEventDataset
from pioneerml.pipelines.training.group_classification_event.steps.config import resolve_step_config
from pioneerml.pipelines.training.group_classification_event.steps.train import (
    _collate_graphs,
    _split_dataset_to_graphs,
)


def _resolve_plot_path(config: dict | None) -> str | None:
    if not config:
        return None
    if "plot_path" in config and config["plot_path"]:
        return str(config["plot_path"])
    if "plot_dir" in config and config["plot_dir"]:
        plot_dir = Path(str(config["plot_dir"]))
        plot_dir.mkdir(parents=True, exist_ok=True)
        return str(plot_dir / "loss_curves.png")
    return None


@step
def evaluate_group_classifier_event(
    module,
    dataset: GroupClassifierEventDataset,
    pipeline_config: dict | None = None,
) -> dict:
    step_config = resolve_step_config(pipeline_config, "evaluate")
    threshold = 0.5 if step_config is None else float(step_config.get("threshold", 0.5))
    module.eval()
    graphs = _split_dataset_to_graphs(dataset)
    if not graphs:
        raise RuntimeError("No graphs available for evaluation.")

    batch_size = int(step_config.get("batch_size", 1)) if step_config else 1
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False, collate_fn=_collate_graphs)

    device = next(module.parameters()).device
    total_loss = 0.0
    total_samples = 0
    preds_all = []
    targets_all = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = module(batch)
            preds = logits[0] if isinstance(logits, (tuple, list)) else logits
            target = batch.y
            if target.dim() == 1 and preds.dim() == 2 and target.numel() % preds.shape[-1] == 0:
                target = target.view(-1, preds.shape[-1])
            loss = module.loss_fn(preds, target)
            bs = int(target.shape[0])
            total_loss += loss.detach().cpu().item() * bs
            total_samples += bs
            preds_all.append(preds.detach().cpu())
            targets_all.append(target.detach().cpu())

    if total_samples == 0:
        raise RuntimeError("No samples available for evaluation.")

    preds_cat = torch.cat(preds_all, dim=0)
    targets_cat = torch.cat(targets_all, dim=0)
    loss = total_loss / total_samples
    probs = torch.sigmoid(preds_cat)
    preds_binary = (probs >= threshold).float()
    accuracy = (preds_binary == targets_cat.float()).float().mean().item()
    exact_match = (preds_binary == targets_cat.float()).all(dim=1).float().mean().item()
    target_int = targets_cat.int()
    preds_int = preds_binary.int()
    num_classes = target_int.shape[-1]
    confusion = torch.zeros((num_classes, 2, 2), dtype=torch.int64)
    for cls_idx in range(num_classes):
        truth = target_int[:, cls_idx]
        pred = preds_int[:, cls_idx]
        tn = ((truth == 0) & (pred == 0)).sum().item()
        fp = ((truth == 0) & (pred == 1)).sum().item()
        fn = ((truth == 1) & (pred == 0)).sum().item()
        tp = ((truth == 1) & (pred == 1)).sum().item()
        confusion[cls_idx, 0, 0] += int(tn)
        confusion[cls_idx, 0, 1] += int(fp)
        confusion[cls_idx, 1, 0] += int(fn)
        confusion[cls_idx, 1, 1] += int(tp)
    confusion_metrics = []
    for cls_idx in range(num_classes):
        tn, fp = confusion[cls_idx, 0].tolist()
        fn, tp = confusion[cls_idx, 1].tolist()
        total = float(tp + fp + fn)
        if total > 0:
            confusion_metrics.append({"tp": tp / total, "fp": fp / total, "fn": fn / total})
        else:
            confusion_metrics.append({"tp": 0.0, "fp": 0.0, "fn": 0.0})

    plot_path = _resolve_plot_path(step_config)
    if plot_path is not None:
        LossCurvesPlot().render(module, save_path=plot_path, show=False)

    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "exact_match": float(exact_match),
        "confusion": confusion_metrics,
        "threshold": float(threshold),
        "train_loss_history": list(module.train_epoch_loss_history),
        "val_loss_history": list(module.val_epoch_loss_history),
        "loss_plot_path": plot_path,
    }
