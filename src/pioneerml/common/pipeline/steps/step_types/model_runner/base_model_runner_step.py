from __future__ import annotations

from ..base_pipeline_step import BasePipelineStep
from .resolvers import ModelRunnerPayloadResolver, ModelRunnerRuntimeConfigResolver


class BaseModelRunnerStep(BasePipelineStep):
    DEFAULT_CONFIG = {
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
                "split_seed": 0,
            },
            "train": {"mode": "train", "shuffle_batches": True, "log_diagnostics": False},
            "val": {"mode": "train", "shuffle_batches": False, "log_diagnostics": False},
            "test": {"mode": "train", "split": "test", "shuffle_batches": False, "log_diagnostics": False},
            "evaluate": {"mode": "train", "shuffle_batches": False, "log_diagnostics": False},
        },
    }
    config_resolver_classes = (ModelRunnerRuntimeConfigResolver,)
    payload_resolver_classes = (ModelRunnerPayloadResolver,)
