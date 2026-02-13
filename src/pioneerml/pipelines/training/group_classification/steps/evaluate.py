import torch
from zenml import step

from pioneerml.common.loader import GroupClassifierGraphLoader
from pioneerml.common.pipeline_utils.evaluation import SimpleClassificationEvaluator
from pioneerml.pipelines.training.group_classification.dataset import GroupClassifierDataset
from pioneerml.pipelines.training.group_classification.steps.config import resolve_step_config

_EVALUATOR = SimpleClassificationEvaluator()


def _evaluate_from_loader(*, module, loader, threshold: float, plot_config: dict | None) -> dict:
    module.eval()
    device = next(module.parameters()).device
    total_loss = 0.0
    total_samples = 0
    preds_all: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            logits = module(batch)
            preds = logits[0] if isinstance(logits, (tuple, list)) else logits
            target = batch.y
            if target.dim() == 1 and preds.dim() == 2 and target.numel() % preds.shape[-1] == 0:
                target = target.view(-1, preds.shape[-1])
            loss = module.loss_fn(preds, target)
            bs = int(target.shape[0])
            total_loss += float(loss.detach().cpu().item()) * bs
            total_samples += bs
            preds_all.append(preds.detach().cpu())
            targets_all.append(target.detach().cpu())

    if total_samples == 0:
        raise RuntimeError("No samples available for evaluation.")

    preds_cat = torch.cat(preds_all, dim=0)
    targets_cat = torch.cat(targets_all, dim=0)
    probs = torch.sigmoid(preds_cat)
    preds_binary = (probs >= float(threshold)).float()
    target_float = targets_cat.float()
    accuracy = float((preds_binary == target_float).float().mean().item())
    exact_match = float((preds_binary == target_float).all(dim=1).float().mean().item())
    loss = float(total_loss / total_samples)

    target_int = targets_cat.int()
    preds_int = preds_binary.int()
    num_classes = target_int.shape[-1]
    confusion_metrics = []
    for cls_idx in range(num_classes):
        truth = target_int[:, cls_idx]
        pred = preds_int[:, cls_idx]
        fp = int(((truth == 0) & (pred == 1)).sum().item())
        fn = int(((truth == 1) & (pred == 0)).sum().item())
        tp = int(((truth == 1) & (pred == 1)).sum().item())
        total = float(tp + fp + fn)
        if total > 0:
            confusion_metrics.append({"tp": tp / total, "fp": fp / total, "fn": fn / total})
        else:
            confusion_metrics.append({"tp": 0.0, "fp": 0.0, "fn": 0.0})

    plot_path = _EVALUATOR.resolve_plot_path(plot_config)
    if plot_path is not None:
        from pioneerml.common.evaluation.plots.loss import LossCurvesPlot

        LossCurvesPlot().render(module, save_path=plot_path, show=False)
    train_history, train_total = _EVALUATOR._concise_history(list(module.train_epoch_loss_history))
    val_history, val_total = _EVALUATOR._concise_history(list(module.val_epoch_loss_history))
    return {
        "loss": loss,
        "accuracy": accuracy,
        "exact_match": exact_match,
        "confusion": confusion_metrics,
        "threshold": float(threshold),
        "train_loss_history": train_history,
        "train_loss_history_total_points": train_total,
        "val_loss_history": val_history,
        "val_loss_history_total_points": val_total,
        "loss_plot_path": plot_path,
    }


@step
def evaluate_group_classifier(
    module,
    dataset: GroupClassifierDataset,
    pipeline_config: dict | None = None,
) -> dict:
    step_config = resolve_step_config(pipeline_config, "evaluate")
    threshold = 0.5 if step_config is None else float(step_config.get("threshold", 0.5))
    batch_size = int(step_config.get("batch_size", 1)) if step_config else 1
    base_loader = getattr(dataset, "loader", None)
    if not isinstance(base_loader, GroupClassifierGraphLoader):
        raise RuntimeError("Dataset is missing GroupClassifierGraphLoader required for chunked evaluation.")
    if not base_loader.include_targets:
        raise RuntimeError("GroupClassifierGraphLoader must run in train mode for evaluation.")
    chunk_row_groups = int(step_config.get("chunk_row_groups", 4)) if step_config else 4
    chunk_workers = int(step_config.get("chunk_workers", 0)) if step_config else 0

    loader_provider = base_loader.with_runtime(
        batch_size=batch_size,
        row_groups_per_chunk=chunk_row_groups,
        num_workers=chunk_workers,
    )
    loader = loader_provider.make_dataloader(shuffle_batches=False)
    return _evaluate_from_loader(module=module, loader=loader, threshold=threshold, plot_config=step_config)
