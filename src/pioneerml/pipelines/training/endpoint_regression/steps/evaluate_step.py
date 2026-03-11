from typing import Any

import torch
from zenml import step

from pioneerml.common.evaluation.evaluators import SimpleRegressionEvaluator
from pioneerml.common.data_loader import BatchBundle
from pioneerml.common.pipeline.steps import BaseEvaluationStep


class EndpointRegressorEvaluateStep(BaseEvaluationStep):
    step_key = "evaluate"

    def default_config(self) -> dict:
        return {
            "plots": ["loss_curves"],
            "batch_size": 64,
            "chunk_row_groups": 4,
            "chunk_workers": None,
        }

    def build_evaluator(self):
        return SimpleRegressionEvaluator()

    def evaluate_from_loader(self, *, loader, cfg: dict[str, Any], module, evaluator) -> dict[str, Any]:
        module.eval()
        device = next(module.parameters()).device
        total_loss = 0.0
        total_mae = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                preds = module(batch)
                preds = preds[0] if isinstance(preds, (tuple, list)) else preds
                target = batch.y_graph
                if target.dim() == 1 and preds.dim() == 2 and target.numel() % preds.shape[-1] == 0:
                    target = target.view(-1, preds.shape[-1])
                loss = module.loss_fn(preds, target)
                mae = torch.abs(preds - target).mean()
                bs = int(target.shape[0])
                total_loss += float(loss.detach().cpu().item()) * bs
                total_mae += float(mae.detach().cpu().item()) * bs
                total_samples += bs

        if total_samples == 0:
            raise RuntimeError("No samples available for evaluation.")

        plot_outputs = self.apply_registered_plots(
            context={
                "plot_kwargs_by_name": {
                    "loss_curves": {
                        "train_losses": module,
                        "save_path": evaluator.resolve_plot_path(cfg),
                        "show": False,
                    }
                }
            },
            plot_names=self.resolve_plot_names(cfg),
        )
        train_history, train_total = evaluator.concise_history(list(module.train_epoch_loss_history))
        val_history, val_total = evaluator.concise_history(list(module.val_epoch_loss_history))
        return {
            "loss": total_loss / total_samples,
            "mae": total_mae / total_samples,
            "train_loss_history": train_history,
            "train_loss_history_total_points": train_total,
            "val_loss_history": val_history,
            "val_loss_history_total_points": val_total,
            "loss_plot_path": plot_outputs.get("loss_curves_path"),
        }


@step(name="evaluate_endpoint_regressor")
def evaluate_endpoint_regressor_step(
    module: Any,
    dataset: BatchBundle,
    pipeline_config: dict | None = None,
) -> dict:
    return EndpointRegressorEvaluateStep(pipeline_config=pipeline_config).execute(
        payloads={"module": module, "loader": dataset}
    )
