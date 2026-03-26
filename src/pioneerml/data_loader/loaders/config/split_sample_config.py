from __future__ import annotations

from dataclasses import dataclass
import random
import warnings


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass
class SplitSampleConfig:
    """Validated split/sampling policy used by row-level filtering."""

    split: str | None = None
    train_fraction: float = 0.9
    val_fraction: float = 0.05
    test_fraction: float = 0.05
    split_seed: int | None = None
    sample_fraction: float | None = None

    def __post_init__(self) -> None:
        split_norm = None if self.split is None else str(self.split).strip().lower()
        if split_norm is not None and split_norm not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {self.split}. Expected one of: 'train', 'val', 'test'.")
        self.split = split_norm

        t = _clamp01(self.train_fraction)
        v = _clamp01(self.val_fraction)
        s = _clamp01(self.test_fraction)
        if (t, v, s) != (float(self.train_fraction), float(self.val_fraction), float(self.test_fraction)):
            warnings.warn(
                "Split fractions were clamped to [0, 1].",
                RuntimeWarning,
                stacklevel=2,
            )

        total = t + v + s
        if total <= 0.0:
            warnings.warn(
                "Invalid split fractions (sum <= 0). Using defaults train=0.9, val=0.05, test=0.05.",
                RuntimeWarning,
                stacklevel=2,
            )
            t, v, s = 0.9, 0.05, 0.05
        elif abs(total - 1.0) > 1e-9:
            remaining = max(0.0, 1.0 - t)
            tail = v + s
            if tail > 0.0:
                scale = remaining / tail
                v *= scale
                s *= scale
            else:
                v, s = 0.0, remaining
            warnings.warn(
                (
                    "Split fractions did not sum to 1.0. "
                    f"Adjusted to train={t:.6f}, val={v:.6f}, test={s:.6f}."
                ),
                RuntimeWarning,
                stacklevel=2,
            )

        self.train_fraction = float(t)
        self.val_fraction = float(v)
        self.test_fraction = float(s)

        if self.sample_fraction is not None:
            sf = _clamp01(self.sample_fraction)
            if sf <= 0.0:
                warnings.warn(
                    "sample_fraction <= 0; disabling sampling filter.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.sample_fraction = None
            else:
                if sf != float(self.sample_fraction):
                    warnings.warn(
                        f"sample_fraction clamped to {sf:.6f}.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                self.sample_fraction = float(sf)

        if self.split_seed is None:
            self.split_seed = int(random.getrandbits(63))

