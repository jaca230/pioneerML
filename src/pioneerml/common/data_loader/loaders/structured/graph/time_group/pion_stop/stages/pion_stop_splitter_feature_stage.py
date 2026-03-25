from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import numpy as np

from pioneerml.common.data_loader.loaders.array_store import NDArrayStore
from pioneerml.common.data_loader.loaders.stage.stages import BaseStage


class PionStopSplitterFeatureStage(BaseStage):
    """Build per-node splitter priors for pion-stop features."""

    name = "build_splitter_features"
    requires = ("layout", "sort_order_out", "node_truth_out")
    provides = ("splitter_probs_out",)

    def __init__(
        self,
        *,
        input_state_key: str = "features_in",
        num_classes: int = 3,
        use_splitter_probs: bool = True,
    ) -> None:
        self.input_state_key = str(input_state_key)
        self.num_classes = int(num_classes)
        self.use_splitter_probs = bool(use_splitter_probs)

    def run_loader(self, *, state: MutableMapping[str, Any], owner) -> None:
        _ = owner
        layout = state["layout"]
        chunk_in = state.get(self.input_state_key)
        if not isinstance(chunk_in, NDArrayStore):
            raise RuntimeError(
                f"Stage '{self.name}' missing required input state map: {self.input_state_key}"
            )

        total_nodes = int(layout["total_nodes"])
        hit_counts = np.asarray(layout["hit_counts"], dtype=np.int64)
        order = np.asarray(state.get("sort_order_out"), dtype=np.int64)
        if int(order.shape[0]) != total_nodes:
            raise RuntimeError("splitter feature stage requires sort_order_out aligned to total_nodes.")

        out = np.zeros((total_nodes, self.num_classes), dtype=np.float32)

        has_splitter_probs = all(
            chunk_in.has_raw(chunk_in.values_key(name)) and chunk_in.has_raw(chunk_in.offsets_key(name, 0))
            for name in ("pred_hit_pion", "pred_hit_muon", "pred_hit_mip")
        )
        if self.use_splitter_probs and has_splitter_probs:
            splitter_counts = (
                chunk_in.offsets("pred_hit_pion", 0)[1:] - chunk_in.offsets("pred_hit_pion", 0)[:-1]
            ).astype(np.int64, copy=False)
            if not np.array_equal(splitter_counts, hit_counts):
                raise RuntimeError("Splitter probability list lengths do not match hit counts.")
            out[:, 0] = chunk_in.values("pred_hit_pion")[order].astype(np.float32, copy=False)
            out[:, 1] = chunk_in.values("pred_hit_muon")[order].astype(np.float32, copy=False)
            out[:, 2] = chunk_in.values("pred_hit_mip")[order].astype(np.float32, copy=False)
        else:
            fallback = state.get("node_truth_out")
            if isinstance(fallback, np.ndarray) and fallback.shape == out.shape:
                out[:] = fallback.astype(np.float32, copy=False)

        state["splitter_probs_out"] = out
