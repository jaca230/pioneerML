from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns

from .base import BasePlot, _to_numpy

try:
    from IPython.display import display  # type: ignore
except Exception:  # pragma: no cover
    display = None


class RegressionDiagnosticsPlot(BasePlot):
    name = "regression_diagnostics"

    def render(
        self,
        predictions: Any,
        targets: Any,
        *,
        save_path: str | Path | None = None,
        show: bool = False,
        component_names: list[str] | None = None,
    ) -> str | None:
        y_true = _to_numpy(targets)
        y_pred = _to_numpy(predictions)
        
        # Handle multi-dimensional outputs (e.g., [N, 2] for angles or [N, 3] for positions)
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            num_components = y_true.shape[1]
            if component_names is None:
                component_names = [f"Component {i}" for i in range(num_components)]
            
            # Create subplots for each component
            fig, axes = plt.subplots(num_components, 2, figsize=(12, 4 * num_components))
            if num_components == 1:
                axes = axes.reshape(1, -1)
            
            for i in range(num_components):
                y_true_comp = y_true[:, i]
                y_pred_comp = y_pred[:, i]
                residuals = y_pred_comp - y_true_comp
                
                # Scatter plot
                axes[i, 0].scatter(y_true_comp, y_pred_comp, alpha=0.6)
                min_val = min(y_true_comp.min(), y_pred_comp.min())
                max_val = max(y_true_comp.max(), y_pred_comp.max())
                axes[i, 0].plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.6)
                axes[i, 0].set_xlabel("True")
                axes[i, 0].set_ylabel("Predicted")
                axes[i, 0].set_title(f"{component_names[i]} - Predicted vs True")
                axes[i, 0].grid(True, linestyle="--", alpha=0.4)
                
                # Residuals
                sns.histplot(residuals, kde=True, ax=axes[i, 1])
                axes[i, 1].axvline(0, color="k", linestyle="--", alpha=0.7)
                axes[i, 1].set_xlabel("Residual")
                axes[i, 1].set_title(f"{component_names[i]} - Residual Distribution")
        else:
            # Single-dimensional output (flatten if needed)
            y_true = y_true.reshape(-1)
            y_pred = y_pred.reshape(-1)
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
                if display is not None:
                    try:
                        display(fig)
                    except Exception:
                        pass
            else:
                plt.show()
        plt.close(fig)
        return save_path
