"""
Training utilities and loops for PIONEER models.

This module will contain:
- Training loops (basic PyTorch and PyTorch Lightning)
- Loss functions
- Optimizers and schedulers
- Callbacks and hooks
- Checkpointing utilities
"""

# Placeholder for future training utilities
# Training functions currently reside in notebooks
# TODO: Extract and modularize training code from notebooks

from pioneerml.common.training.lightning import GraphLightningModule
from pioneerml.common.training.compile import maybe_compile_model, restore_eager_model_if_compiled
from pioneerml.common.training.early_stopping import build_early_stopping_callback
from pioneerml.common.training.utils import set_tensor_core_precision, default_precision_for_accelerator

__all__ = [
    "GraphLightningModule",
    "maybe_compile_model",
    "restore_eager_model_if_compiled",
    "build_early_stopping_callback",
    "set_tensor_core_precision",
    "default_precision_for_accelerator",
]
