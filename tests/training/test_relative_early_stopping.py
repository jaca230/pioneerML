from __future__ import annotations

import torch

from pioneerml.common.pipeline.services.training.utils import RelativeEarlyStopping


def test_relative_early_stopping_min_mode_requires_actual_relative_improvement():
    cb = RelativeEarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=3,
        min_delta=0.0,
        relative_min_delta=0.05,
    )

    # Establish baseline best.
    stop, _ = cb._evaluate_stopping_criteria(torch.tensor(1.0))
    assert stop is False
    assert float(cb.best_score) == 1.0

    # Large improvement >5%: updates best.
    stop, _ = cb._evaluate_stopping_criteria(torch.tensor(0.5))
    assert stop is False
    assert float(cb.best_score) == 0.5
    assert cb.wait_count == 0

    # 0.49 is only 2% better than 0.5, so should NOT reset patience.
    stop, _ = cb._evaluate_stopping_criteria(torch.tensor(0.49))
    assert stop is False
    assert float(cb.best_score) == 0.5
    assert cb.wait_count == 1

    # Still no >=5% relative improvement.
    stop, _ = cb._evaluate_stopping_criteria(torch.tensor(0.495))
    assert stop is False
    assert cb.wait_count == 2

    # Patience exhausted.
    stop, _ = cb._evaluate_stopping_criteria(torch.tensor(0.50))
    assert stop is True


def test_relative_early_stopping_max_mode_requires_actual_relative_improvement():
    cb = RelativeEarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=2,
        min_delta=0.0,
        relative_min_delta=0.10,
    )

    stop, _ = cb._evaluate_stopping_criteria(torch.tensor(0.50))
    assert stop is False
    assert float(cb.best_score) == 0.50

    # +6% is below 10% threshold, so no improvement.
    stop, _ = cb._evaluate_stopping_criteria(torch.tensor(0.53))
    assert stop is False
    assert float(cb.best_score) == 0.50
    assert cb.wait_count == 1

    # Patience exhausted on next non-improving value.
    stop, _ = cb._evaluate_stopping_criteria(torch.tensor(0.54))
    assert stop is True
