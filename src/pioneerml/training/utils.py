"""
Training utilities for device configuration and defaults.
"""

from __future__ import annotations

import torch


def set_tensor_core_precision(mode: str = "medium") -> str | None:
    """
    Configure matmul precision to leverage tensor cores on supported GPUs.

    Args:
        mode: One of {"high", "medium"} accepted by torch.set_float32_matmul_precision.

    Returns:
        The mode applied, or None if CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return None
    if mode not in {"high", "medium"}:
        mode = "medium"
    torch.set_float32_matmul_precision(mode)
    return mode


def default_precision_for_accelerator(accelerator: str | None) -> str:
    """
    Pick a Trainer precision string based on accelerator.

    Returns:
        "16-mixed" for CUDA/auto, otherwise "32-true".
    """
    if accelerator in {"cuda", "gpu", "auto"} and torch.cuda.is_available():
        return "16-mixed"
    return "32-true"
