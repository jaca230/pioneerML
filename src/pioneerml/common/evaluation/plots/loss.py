from __future__ import annotations

import math
import numbers
from typing import Iterable, Optional

import matplotlib.pyplot as plt

from .base_plot import BasePlot
from .registry import REGISTRY as PLOT_REGISTRY_DEF

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


@PLOT_REGISTRY_DEF.register("loss_curves")
class LossCurvesPlot(BasePlot):
    name = "loss_curves"

    @staticmethod
    def _resolve_val_accuracy_history(
        *,
        train_losses_obj: object,
        train_hist: list[float],
        val_hist: list[float],
        val_accuracy: Iterable[float] | float | None,
    ) -> list[float]:
        raw = val_accuracy
        if raw is None and hasattr(train_losses_obj, "val_accuracy_history"):
            raw = getattr(train_losses_obj, "val_accuracy_history")
        if raw is None:
            return []

        if isinstance(raw, numbers.Real):
            n = len(val_hist) if len(val_hist) > 0 else len(train_hist)
            if n <= 0:
                n = 1
            return [float(raw)] * int(n)

        out = [float(v) for v in list(raw)]
        while len(out) > len(val_hist) and len(val_hist) > 0:
            out = out[1:]
        return out

    @staticmethod
    def _can_use_log_scale(values: Iterable[float]) -> bool:
        vals = [float(v) for v in values]
        if not vals:
            return False
        return all(math.isfinite(v) and v > 0.0 for v in vals)

    def render(
        self,
        train_losses: Iterable[float] | object,
        val_losses: Optional[Iterable[float]] = None,
        val_accuracy: Optional[Iterable[float] | float] = None,
        *,
        title: str = "Loss Curves",
        xlabel: str = "Epoch",
        log_scale: bool = True,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> str | None:
        train_hist, val_hist = _resolve_histories(train_losses, val_losses)
        val_acc_hist = self._resolve_val_accuracy_history(
            train_losses_obj=train_losses,
            train_hist=train_hist,
            val_hist=val_hist,
            val_accuracy=val_accuracy,
        )

        fig, ax_train = plt.subplots(figsize=(6, 4))
        ax_secondary = None

        if train_hist and val_hist:
            ax_train.plot(train_hist, color="tab:blue", label="train_loss")
            ax_secondary = ax_train.twinx()
            ax_secondary.plot(val_hist, color="tab:orange", label="val_loss")
            ax_train.set_ylabel("Train Loss")
            ax_secondary.set_ylabel("Validation Loss")
        else:
            if train_hist:
                ax_train.plot(train_hist, color="tab:blue", label="train_loss")
            if val_hist:
                ax_train.plot(val_hist, color="tab:orange", label="val_loss")
            ax_train.set_ylabel("Loss")

        # Optional fallback: if no val loss history is available, render validation
        # accuracy on the secondary axis when provided.
        if ax_secondary is None and val_acc_hist:
            ax_secondary = ax_train.twinx()
            x_vals = list(range(len(val_acc_hist)))
            ax_secondary.plot(x_vals, val_acc_hist, color="tab:green", linestyle="--", label="val_accuracy")
            ax_secondary.set_ylabel("Validation Accuracy")
            acc_min = min(val_acc_hist)
            acc_max = max(val_acc_hist)
            if 0.0 <= acc_min and acc_max <= 1.0:
                ax_secondary.set_ylim(0.0, 1.0)

        if log_scale:
            if self._can_use_log_scale(train_hist):
                ax_train.set_yscale("log")
            if ax_secondary is not None and val_hist and self._can_use_log_scale(val_hist):
                ax_secondary.set_yscale("log")

        ax_train.set_title(title)
        ax_train.set_xlabel(xlabel)

        handles, labels = ax_train.get_legend_handles_labels()
        if ax_secondary is not None:
            h2, l2 = ax_secondary.get_legend_handles_labels()
            handles += h2
            labels += l2
        if handles:
            ax_train.legend(handles, labels)
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
