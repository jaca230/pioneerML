"""
Simple plotting utilities for training diagnostics.
"""

from __future__ import annotations

from typing import Iterable, Optional

import matplotlib.pyplot as plt


def plot_loss_curves(
    train_losses: Iterable[float],
    val_losses: Optional[Iterable[float]] = None,
    *,
    title: str = "Loss Curves",
    xlabel: str = "Step",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Plot training/validation loss histories.

    Args:
        train_losses: Sequence of training loss values.
        val_losses: Optional sequence of validation loss values.
        title: Plot title.
        xlabel: Label for the x-axis (e.g., "Step" or "Epoch").
        save_path: If provided, save the figure to this path.
        show: If True, display the plot (useful in notebooks).
    """
    plt.figure(figsize=(6, 4))
    plt.plot(list(train_losses), label="train_loss")
    if val_losses is not None:
        plt.plot(list(val_losses), label="val_loss")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()
