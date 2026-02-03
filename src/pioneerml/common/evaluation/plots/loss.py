from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt

from .base import BasePlot

try:
    from IPython.display import display  # type: ignore
except Exception:  # pragma: no cover - optional
    display = None


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


class LossCurvesPlot(BasePlot):
    name = "loss_curves"

    def render(
        self,
        train_losses: Iterable[float] | object,
        val_losses: Optional[Iterable[float]] = None,
        *,
        title: str = "Loss Curves",
        xlabel: str = "Epoch",
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> str | None:
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

        # Save
        if save_path is not None:
            save_path = str(save_path)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        # Show
        if show:
            backend = plt.get_backend().lower()
            if backend.startswith("agg"):
                if display is not None:
                    try:
                        display(fig)
                    except Exception:
                        pass
            else:
                plt.show()

        plt.close(fig)
        return save_path
