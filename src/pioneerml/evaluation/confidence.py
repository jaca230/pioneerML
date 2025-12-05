"""
Confidence/uncertainty analysis plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .plots import _prepare_classification_inputs, register_plot


@register_plot("confidence_analysis")
def plot_confidence_analysis(
    predictions: Any,
    targets: Any,
    *,
    class_names: Sequence[str] | None = None,
    bins: int = 25,
    save_path: str | Path | None = None,
    show: bool = False,
) -> str | None:
    """
    Plot calibration-style confidence vs. accuracy for each class.
    """
    y_true_binary, y_score, labels, _, num_classes = _prepare_classification_inputs(
        predictions, targets, class_names
    )

    fig, axes = plt.subplots(1, num_classes, figsize=(4 * num_classes, 4))
    axes = np.atleast_1d(axes)

    for idx in range(num_classes):
        ax = axes[idx]
        probs = y_score[:, idx]
        truths = y_true_binary[:, idx]

        bin_edges = np.linspace(0, 1, bins + 1)
        bin_ids = np.digitize(probs, bin_edges) - 1

        bin_acc = []
        bin_conf = []
        for b in range(bins):
            mask = bin_ids == b
            if not np.any(mask):
                continue
            bin_conf.append(probs[mask].mean())
            bin_acc.append(truths[mask].mean())

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly calibrated")
        ax.plot(bin_conf, bin_acc, marker="o", label="Empirical")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Empirical accuracy")
        ax.set_title(f"{labels[idx]} (N={len(probs)})")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

    plt.tight_layout()
    if save_path is not None:
        save_path = str(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        backend = plt.get_backend().lower()
        if backend.startswith("agg"):
            try:
                from IPython.display import display
                display(fig)
            except Exception:
                pass
        else:
            plt.show()
    plt.close(fig)
    return save_path
