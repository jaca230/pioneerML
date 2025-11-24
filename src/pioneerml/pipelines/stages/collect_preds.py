"""
Stage to generate and cache predictions/targets from a dataloader.

Designed to be reusable in pipelines so notebooks don't need ad-hoc collection code.
"""

from __future__ import annotations

from typing import Any

from pioneerml.evaluation import resolve_preds_targets
from pioneerml.pipelines.stage import Stage, StageConfig


class CollectPredsStage(Stage):
    """
    Run a dataloader and store predictions/targets in the pipeline context.

    Params (config.params):
        dataloader: Which dataloader to use from the datamodule ("val" by default).
        module_key: Context key for the model/lightning module (default: "lightning_module").
        datamodule_key: Context key for the datamodule (default: "datamodule").
        preds_key / targets_key: Keys to store results in the context (default: "preds"/"targets").
    """

    def execute(self, context: Any) -> None:
        params = self.config.params
        dataloader = params.get("dataloader", "val")
        preds_key = params.get("preds_key", "preds")
        targets_key = params.get("targets_key", "targets")

        module_key = params.get("module_key", "lightning_module")
        datamodule_key = params.get("datamodule_key", "datamodule")
        if module_key not in context or datamodule_key not in context:
            raise KeyError(
                f"CollectPredsStage requires '{module_key}' and '{datamodule_key}' in context. "
                "Run training or provide these objects first."
            )

        preds, targets = resolve_preds_targets(context, dataloader=dataloader)
        context[preds_key] = preds
        context[targets_key] = targets
