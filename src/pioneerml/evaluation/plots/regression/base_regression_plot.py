from __future__ import annotations

from typing import Any

import numpy as np

from ..base_plot import BasePlot, _to_numpy


class RegressionPlotBase(BasePlot):
    """Base class for regression plots with aligned numeric arrays."""

    def prepare_regression_inputs(
        self,
        *,
        predictions: Any,
        targets: Any,
        allow_flatten: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        y_true = _to_numpy(targets)
        y_pred = _to_numpy(predictions)

        if allow_flatten:
            y_true = np.asarray(y_true).reshape(-1)
            y_pred = np.asarray(y_pred).reshape(-1)
        else:
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Predictions and targets must have identical shapes; got {y_pred.shape} vs {y_true.shape}."
            )
        if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
            raise ValueError("Predictions and targets contain NaN or inf values.")
        return y_pred, y_true

