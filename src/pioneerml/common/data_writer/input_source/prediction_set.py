from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PredictionSet:
    """Aligned prediction payload for one source chunk."""

    src_path: Path
    prediction_event_ids_np: np.ndarray
    model_outputs_by_name: dict[str, np.ndarray]
    num_rows: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "src_path", Path(self.src_path).expanduser().resolve())
        object.__setattr__(self, "prediction_event_ids_np", np.asarray(self.prediction_event_ids_np, dtype=np.int64))
        object.__setattr__(self, "num_rows", int(self.num_rows))
        normalized_outputs = {
            str(name): np.asarray(values)
            for name, values in dict(self.model_outputs_by_name).items()
        }
        object.__setattr__(self, "model_outputs_by_name", normalized_outputs)
        self.validate()

    def validate(self) -> None:
        n = int(self.prediction_event_ids_np.shape[0])
        for key, values in self.model_outputs_by_name.items():
            arr = np.asarray(values)
            if arr.ndim < 1:
                raise ValueError(f"Model output '{key}' must have at least 1 dimension, got shape {tuple(arr.shape)}.")
            if int(arr.shape[0]) != n:
                raise ValueError(
                    f"Model output '{key}' leading dim {arr.shape[0]} does not match prediction_event_ids length {n}."
                )
