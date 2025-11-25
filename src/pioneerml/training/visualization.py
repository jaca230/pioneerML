"""
Plotting utilities for training diagnostics.
"""

from __future__ import annotations
from typing import Iterable, Optional

import matplotlib.pyplot as plt


def _resolve_histories(train_losses, val_losses=None):
    """Accept either explicit loss arrays or a LightningModule with stored histories."""
    if hasattr(train_losses, "train_epoch_loss_history"):
        module = train_losses
        train_losses = (
            getattr(module, "train_epoch_loss_history", None)
            or getattr(module, "train_loss_history", None)
        )
        val_losses = (
            getattr(module, "val_epoch_loss_history", None)
            or getattr(module, "val_loss_history", None)
        )

    train_hist = list(train_losses) if train_losses is not None else []
    val_hist = list(val_losses) if val_losses is not None else []

    # Lightning runs a val sanity check before the first train epoch; trim any
    # leading val entries so lengths align with train epochs.
    while len(val_hist) > len(train_hist) and len(train_hist) > 0:
        val_hist = val_hist[1:]
    return train_hist, val_hist


def plot_loss_curves(
    train_losses: Iterable[float] | object,
    val_losses: Optional[Iterable[float]] = None,
    *,
    title: str = "Loss Curves",
    xlabel: str = "Epoch",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Plot training/validation loss histories.
    """
    train_hist, val_hist = _resolve_histories(train_losses, val_losses)

    fig, ax = plt.subplots(figsize=(6, 4))

    if train_hist:
        ax.plot(train_hist, label="train_loss")
    if val_hist:
        ax.plot(val_hist, label="val_loss")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        backend = plt.get_backend().lower()
        if backend.startswith("agg"):
            try:
                from IPython.display import display
                display(fig)
            except Exception:
                # Fall back silently when display is unavailable
                pass
        else:
            plt.show()

    plt.close(fig)
    return fig
