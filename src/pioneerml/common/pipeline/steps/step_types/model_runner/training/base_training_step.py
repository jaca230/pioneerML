from __future__ import annotations

from ..base_model_runner_step import BaseModelRunnerStep
from .resolvers import TrainingRuntimeConfigResolver
from .utils import LightningWarningFilter
from ..utils import merge_nested_dicts


class BaseTrainingStep(BaseModelRunnerStep):
    DEFAULT_CONFIG = merge_nested_dicts(
        base=BaseModelRunnerStep.DEFAULT_CONFIG,
        override={
            "max_epochs": 10,
            "grad_clip": 2.0,
            "trainer_kwargs": {"enable_progress_bar": True},
            "early_stopping": {
                "enabled": True,
                "type": "relative",
                "config": {
                    "monitor": "val_loss",
                    "mode": "min",
                    "patience": 5,
                    "min_delta": 0.05,
                    "strict": True,
                    "check_finite": True,
                    "verbose": False,
                },
            },
        },
    )
    config_resolver_classes = BaseModelRunnerStep.config_resolver_classes + (TrainingRuntimeConfigResolver,)
    payload_resolver_classes = BaseModelRunnerStep.payload_resolver_classes

    def __init__(self, *, pipeline_config: dict | None = None) -> None:
        super().__init__(pipeline_config=pipeline_config)
        self._warning_filter = LightningWarningFilter()

    def apply_warning_filter(self) -> None:
        self._warning_filter.apply_default()
