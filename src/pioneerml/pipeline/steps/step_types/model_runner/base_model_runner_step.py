from __future__ import annotations

from ..base_pipeline_step import BasePipelineStep
from .resolvers import ModelRunnerStateResolver, ModelRunnerConfigResolver


class BaseModelRunnerStep(BasePipelineStep):
    DEFAULT_CONFIG = {
        "compiler": {
            "type": "torch_compile",
            "config": {
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
        },
        "loader_manager": {
            "type": "config",
            "config": {
                "input_sources_spec": {
                    "main_sources": [],
                    "optional_sources_by_name": {},
                    "source_type": "file",
                },
                "input_backend": {
                    "type": "parquet",
                    "config": {},
                },
                "defaults": {
                    "type": "group_classifier",
                    "config": {
                        "batch_size": None,
                        "mode": "train",
                        "chunk_row_groups": 4,
                        "chunk_workers": None,
                        "edge_template_cache_enabled": False,
                        "edge_template_cache_max_entries": None,
                        "sample_fraction": 1.0,
                        "train_fraction": 0.80,
                        "val_fraction": 0.10,
                        "test_fraction": 0.10,
                        "split_seed": 0,
                    },
                },
                "loaders": {},
            },
        },
    }
    config_resolver_classes = (ModelRunnerConfigResolver,)
    payload_resolver_classes = (ModelRunnerStateResolver,)
