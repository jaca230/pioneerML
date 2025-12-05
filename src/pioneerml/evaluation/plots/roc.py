from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from .base import BasePlot, _prepare_classification_inputs

try:
    from IPython.display import display  # type: ignore
except Exception:  # pragma: no cover
    display = None


class RocCurvesPlot(BasePlot):
    name = "roc"

    def render(
        self,
        predictions: Any,
        targets: Any,
        *,
        class_names: Sequence[str] | None = None,
        save_path: str | Path | None = None,
        show: bool = False,
    ) -> str | None:
        y_true_binary, y_score, labels, _, num_classes = _prepare_classification_inputs(
            predictions, targets, class_names
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        for idx in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_binary[:, idx], y_score[:, idx])
            try:
                auc = roc_auc_score(y_true_binary[:, idx], y_score[:, idx])
                label = f"{labels[idx]} (AUC={auc:.3f})"
            except ValueError:
                label = labels[idx]
            ax.plot(fpr, tpr, label=label)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
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
