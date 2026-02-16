from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping, EarlyStoppingReason


class RelativeEarlyStopping(EarlyStopping):
    """Early stopping that interprets min_delta as a relative threshold."""

    def __init__(self, *args: Any, relative_min_delta: float = 0.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.relative_min_delta = max(0.0, float(relative_min_delta))

    def _evaluate_stopping_criteria(self, current: torch.Tensor) -> tuple[bool, str | None]:
        # Keep base behavior for finite/threshold/divergence checks.
        if self.check_finite and not torch.isfinite(current):
            return super()._evaluate_stopping_criteria(current)
        if self.stopping_threshold is not None and self.monitor_op(current, self.stopping_threshold):
            return super()._evaluate_stopping_criteria(current)
        if self.divergence_threshold is not None and self.monitor_op(-current, -self.divergence_threshold):
            return super()._evaluate_stopping_criteria(current)

        rel_delta_abs = torch.abs(self.best_score).to(current.device) * self.relative_min_delta
        improved = self.monitor_op(current - rel_delta_abs, self.best_score.to(current.device))
        if improved:
            self.wait_count = 0
            self.best_score = current
            return False, self._improvement_message(current)

        self.wait_count += 1
        if self.wait_count >= self.patience:
            self.stopping_reason = EarlyStoppingReason.PATIENCE_EXHAUSTED
            reason = (
                f"Monitored metric {self.monitor} did not improve in the last {self.wait_count} records."
                f" Best score: {self.best_score:.3f}. Signaling Trainer to stop."
            )
            return True, reason
        return False, None


def build_early_stopping_callback(cfg: dict | None) -> pl.callbacks.EarlyStopping | None:
    config = dict(cfg or {})
    if not bool(config.get("enabled", False)):
        return None

    monitor = str(config.get("monitor", "val_loss"))
    mode = str(config.get("mode", "min"))
    patience = int(config.get("patience", 5))
    min_delta = float(config.get("min_delta", 0.0))
    strict = bool(config.get("strict", True))
    check_finite = bool(config.get("check_finite", True))
    verbose = bool(config.get("verbose", False))
    delta_mode = str(config.get("min_delta_mode", "absolute")).strip().lower()

    common_kwargs = {
        "monitor": monitor,
        "mode": mode,
        "patience": patience,
        "strict": strict,
        "check_finite": check_finite,
        "verbose": verbose,
    }
    if delta_mode in {"relative", "percent", "pct"}:
        return RelativeEarlyStopping(relative_min_delta=min_delta, min_delta=0.0, **common_kwargs)
    return pl.callbacks.EarlyStopping(min_delta=min_delta, **common_kwargs)
