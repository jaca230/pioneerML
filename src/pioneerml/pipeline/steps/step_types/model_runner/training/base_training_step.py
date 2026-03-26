from __future__ import annotations

from pioneerml.logging.filters import BaseLogFilter, LogFilterFactory

from ..base_model_runner_step import BaseModelRunnerStep
from .resolvers import TrainingConfigResolver
from ..utils import merge_nested_dicts


class BaseTrainingStep(BaseModelRunnerStep):
    DEFAULT_CONFIG = merge_nested_dicts(
        base=BaseModelRunnerStep.DEFAULT_CONFIG,
        override={
            "architecture": {
                "type": "required",
                "config": {},
            },
            "module": {
                "type": "graph_lightning",
                "config": {},
            },
            "trainer": {
                "type": "lightning_module",
                "config": {
                    "trainer_kwargs": {
                        "enable_progress_bar": True,
                        "max_epochs": 10,
                        "gradient_clip_val": 2.0,
                    },
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
            },
            "log_filter": {
                "type": "training",
                "config": {},
            },
            "loader_manager": {
                "config": {
                    "loaders": {
                        "train_loader": {
                            "config": {"mode": "train", "shuffle_batches": True, "log_diagnostics": False},
                        },
                        "val_loader": {
                            "config": {"mode": "train", "shuffle_batches": False, "log_diagnostics": False},
                        },
                    },
                },
            },
        },
    )
    config_resolver_classes = BaseModelRunnerStep.config_resolver_classes + (TrainingConfigResolver,)
    payload_resolver_classes = BaseModelRunnerStep.payload_resolver_classes

    def __init__(self, *, pipeline_config: dict | None = None) -> None:
        super().__init__(pipeline_config=pipeline_config)
        log_filter_block = dict(self.config_json.get("log_filter") or {})
        log_filter_name = str(log_filter_block["type"]).strip()
        log_filter_cfg = dict(log_filter_block.get("config") or {})
        self._log_filter: BaseLogFilter = LogFilterFactory(log_filter_name=log_filter_name).build(config=log_filter_cfg)

    def apply_warning_filter(self) -> None:
        self._log_filter.apply()
