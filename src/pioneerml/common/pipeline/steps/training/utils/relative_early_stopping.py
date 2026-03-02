from __future__ import annotations

import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping, EarlyStoppingReason


class RelativeEarlyStopping(EarlyStopping):
    """Early stopping callback using relative improvement threshold."""

    def __init__(self, *args, relative_min_delta: float = 0.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.relative_min_delta = max(0.0, float(relative_min_delta))

    def _evaluate_stopping_criteria(self, current: torch.Tensor) -> tuple[bool, str | None]:
        if self.check_finite and not torch.isfinite(current):
            return super()._evaluate_stopping_criteria(current)
        if self.stopping_threshold is not None and self.monitor_op(current, self.stopping_threshold):
            return super()._evaluate_stopping_criteria(current)
        if self.divergence_threshold is not None and self.monitor_op(-current, -self.divergence_threshold):
            return super()._evaluate_stopping_criteria(current)

        best = self.best_score.to(current.device)
        if not torch.isfinite(best):
            improved = True
        else:
            rel_delta_abs = torch.abs(best) * self.relative_min_delta
            if self.mode == "min":
                improved = current < (best - rel_delta_abs)
            else:
                improved = current > (best + rel_delta_abs)
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
