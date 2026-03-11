from __future__ import annotations

from ..base_pipeline_step import BasePipelineStep
from .resolvers import TrainingRuntimeConfigResolver, TrainingRuntimeStateResolver
from .utils import LightningWarningFilter


class BaseTrainingStep(BasePipelineStep):
    DEFAULT_CONFIG = {
        "max_epochs": 10,
        "grad_clip": 2.0,
        "trainer_kwargs": {"enable_progress_bar": True},
        "compile": {
            "enabled": False,
            "mode": "default",
            "dynamic": None,
            "backend": None,
            "matmul_precision": "high",
            "capture_scalar_outputs": True,
            "cudagraph_skip_dynamic_graphs": True,
            "max_autotune": False,
            "max_autotune_gemm": False,
            "inductor_log_level": "ERROR",
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
        "loader_config": {
            "base": {
                "batch_size": 64,
                "mode": "train",
                "chunk_row_groups": 4,
                "chunk_workers": None,
                "sample_fraction": 1.0,
                "train_fraction": 0.80,
                "val_fraction": 0.10,
                "test_fraction": 0.10,
                "split_seed": None,
            },
            "train": {"mode": "train", "shuffle_batches": True, "log_diagnostics": False},
            "val": {"mode": "train", "shuffle_batches": False, "log_diagnostics": False},
        },
    }
    config_resolver_classes = (TrainingRuntimeConfigResolver,)
    payload_resolver_classes = (TrainingRuntimeStateResolver,)

    def __init__(self, *, pipeline_config: dict | None = None) -> None:
        super().__init__(pipeline_config=pipeline_config)
        self._warning_filter = LightningWarningFilter()

    def apply_warning_filter(self) -> None:
        self._warning_filter.apply_default()
