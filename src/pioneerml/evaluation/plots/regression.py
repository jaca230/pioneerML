from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns

from .base import BasePlot, _to_numpy


class RegressionDiagnosticsPlot(BasePlot):
    name = "regression_diagnostics"

    def render(
        self,
        predictions: Any,
        targets: Any,
        *,
        save_path: str | Path | None = None,
        show: bool = False,
    ) -> str | None:
        y_true = _to_numpy(targets).reshape(-1)
        y_pred = _to_numpy(predictions).reshape(-1)
        residuals = y_pred - y_true

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.6)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.6)
        axes[0].set_xlabel("True")
        axes[0].set_ylabel("Predicted")
        axes[0].set_title("Predicted vs True")
        axes[0].grid(True, linestyle="--", alpha=0.4)

        # Residuals
        sns.histplot(residuals, kde=True, ax=axes[1])
        axes[1].axvline(0, color="k", linestyle="--", alpha=0.7)
        axes[1].set_xlabel("Residual")
        axes[1].set_title("Residual Distribution")

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
