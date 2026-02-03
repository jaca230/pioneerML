from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve

from .base import BasePlot, _prepare_classification_inputs

try:
    from IPython.display import display  # type: ignore
except Exception:  # pragma: no cover
    display = None


class PrecisionRecallPlot(BasePlot):
    name = "precision_recall"

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
            precision, recall, _ = precision_recall_curve(y_true_binary[:, idx], y_score[:, idx])
            ap = average_precision_score(y_true_binary[:, idx], y_score[:, idx])
            ax.plot(recall, precision, label=f"{labels[idx]} (AP={ap:.3f})")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves")
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
