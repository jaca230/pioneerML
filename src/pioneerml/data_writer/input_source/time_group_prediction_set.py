from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .prediction_set import PredictionSet


@dataclass(frozen=True)
class TimeGroupPredictionSet(PredictionSet):
    """Prediction payload with explicit time-group alignment metadata."""

    time_group_event_ids_np: np.ndarray
    time_group_ids_np: np.ndarray
    group_id_column: str = "time_group_ids"

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "time_group_event_ids_np", np.asarray(self.time_group_event_ids_np, dtype=np.int64))
        object.__setattr__(self, "time_group_ids_np", np.asarray(self.time_group_ids_np, dtype=np.int64))
        object.__setattr__(self, "group_id_column", str(self.group_id_column))
        self.validate()

    def validate(self) -> None:
        super().validate()
        if int(self.time_group_event_ids_np.shape[0]) != int(self.time_group_ids_np.shape[0]):
            raise ValueError(
                "time_group_event_ids_np and time_group_ids_np must have matching lengths, got "
                f"{self.time_group_event_ids_np.shape[0]} vs {self.time_group_ids_np.shape[0]}."
            )

