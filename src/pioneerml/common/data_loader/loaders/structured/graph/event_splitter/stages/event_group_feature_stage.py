from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import numpy as np

from pioneerml.common.data_loader.loaders.array_store import NDArrayStore
from pioneerml.common.data_loader.loaders.stage.stages import BaseStage


class EventGroupFeatureStage(BaseStage):
    """Build event-splitter per-group classifier priors."""

    name = "build_group_features"
    requires = ("layout",)
    provides = ("group_probs_out",)

    def __init__(
        self,
        *,
        input_state_key: str = "features_in",
        num_classes: int = 3,
        use_group_probs: bool = True,
    ) -> None:
        self.input_state_key = str(input_state_key)
        self.num_classes = int(num_classes)
        self.use_group_probs = bool(use_group_probs)

    def run_loader(self, *, state: MutableMapping[str, Any], owner) -> None:
        _ = owner
        layout = state["layout"]
        chunk_in = state.get(self.input_state_key)
        if not isinstance(chunk_in, NDArrayStore):
            raise RuntimeError(
                f"Stage '{self.name}' missing required input state map: {self.input_state_key}"
            )
        total_groups = int(layout["total_groups"])
        row_group_counts = np.asarray(layout["row_group_counts"], dtype=np.int64)

        group_probs_out = np.zeros((total_groups, self.num_classes), dtype=np.float32)
        has_group_probs = all(
            chunk_in.has_raw(chunk_in.values_key(name)) and chunk_in.has_raw(chunk_in.offsets_key(name, 0))
            for name in ("pred_pion", "pred_muon", "pred_mip")
        )

        if self.use_group_probs and has_group_probs:
            pred_counts = (chunk_in.offsets("pred_pion", 0)[1:] - chunk_in.offsets("pred_pion", 0)[:-1]).astype(
                np.int64, copy=False
            )
            if not np.array_equal(pred_counts, row_group_counts):
                raise RuntimeError("pred_pion list lengths do not match inferred row group counts.")
            pred_pion = chunk_in.values("pred_pion").astype(np.float32, copy=False)
            pred_muon = chunk_in.values("pred_muon").astype(np.float32, copy=False)
            pred_mip = chunk_in.values("pred_mip").astype(np.float32, copy=False)
            if (
                int(pred_pion.shape[0]) != total_groups
                or int(pred_muon.shape[0]) != total_groups
                or int(pred_mip.shape[0]) != total_groups
            ):
                raise RuntimeError("Group probability flattened lengths must match total inferred groups.")
            group_probs_out[:, 0] = pred_pion
            group_probs_out[:, 1] = pred_muon
            group_probs_out[:, 2] = pred_mip
        else:
            fallback = state.get("group_truth_out")
            if isinstance(fallback, np.ndarray) and fallback.shape == group_probs_out.shape:
                group_probs_out[:] = fallback.astype(np.float32, copy=False)

        state["group_probs_out"] = group_probs_out
