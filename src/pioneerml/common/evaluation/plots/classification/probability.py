from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..registry import register_plot
from .base_classification_plot import ClassificationPlotBase

try:
    from IPython.display import display  # type: ignore
except Exception:  # pragma: no cover
    display = None


@register_plot("probability_distributions")
class ProbabilityDistributionsPlot(ClassificationPlotBase):
    name = "probability_distributions"

    def render(
        self,
        predictions: Any,
        targets: Any,
        *,
        class_names: Sequence[str] | None = None,
        bins: int = 25,
        save_path: str | Path | None = None,
        show: bool = False,
    ) -> str | None:
        y_true_binary, y_score, labels, _, num_classes = self.prepare_classification_inputs(
            predictions=predictions,
            targets=targets,
            class_names=class_names,
        )

        fig, axes = plt.subplots(1, num_classes, figsize=(4 * num_classes, 4))
        axes = np.atleast_1d(axes)

        for idx in range(num_classes):
            ax = axes[idx]
            ax.hist(y_score[:, idx], bins=bins, alpha=0.7, color="tab:blue")
            ax.set_yscale("log")
            ax.set_title(f"{labels[idx]} (N={len(y_score)})")
            ax.set_xlabel("Predicted probability")
            ax.set_ylabel("Count (log scale)")

        plt.tight_layout()
        if save_path is not None:
            save_path = str(save_path)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
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
