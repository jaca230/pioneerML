from pathlib import Path

import torch
from zenml import step

from pioneerml.evaluation.plots.loss import LossCurvesPlot

from pioneerml.zenml.pipelines.training.group_classification.batch import GroupClassifierBatch


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
def evaluate_group_classifier(
    module,
    batch: GroupClassifierBatch,
    *,
    step_config: dict | None = None,
) -> dict:
    module.eval()
    data = batch.data.to(next(module.parameters()).device)
    with torch.no_grad():
        logits = module(data)
        loss = module.loss_fn(logits, batch.targets.to(logits.device)).detach().cpu().item()
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        accuracy = (preds == batch.targets.to(preds.device)).float().mean().detach().cpu().item()

    plot_path = _resolve_plot_path(step_config)
    if plot_path is not None:
        LossCurvesPlot().render(module, save_path=plot_path, show=False)

    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "train_loss_history": list(module.train_epoch_loss_history),
        "val_loss_history": list(module.val_epoch_loss_history),
        "loss_plot_path": plot_path,
    }
