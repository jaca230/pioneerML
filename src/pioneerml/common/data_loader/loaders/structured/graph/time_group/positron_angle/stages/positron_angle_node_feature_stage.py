from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import numpy as np

from pioneerml.common.data_loader.loaders.array_store import NDArrayStore
from pioneerml.common.data_loader.loaders.structured.graph.time_group.pion_stop.stages.pion_stop_node_feature_stage import (
    PionStopNodeFeatureStage,
)


class PositronAngleNodeFeatureStage(PionStopNodeFeatureStage):
    """Build positron-angle node features and expose sorted PDG IDs for filtering."""

    provides = PionStopNodeFeatureStage.provides + ("sorted_pdg_id_out", "has_pdg_id_out")

    def __init__(
        self,
        *,
        pdg_id_field: str = "hits_pdg_id",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.pdg_id_field = str(pdg_id_field)

    def run_loader(self, *, state: MutableMapping[str, Any], owner) -> None:
        super().run_loader(state=state, owner=owner)

        chunk_in = state.get(self.input_state_key)
        if not isinstance(chunk_in, NDArrayStore):
            raise RuntimeError(
                f"Stage '{self.name}' missing required input state map: {self.input_state_key}"
            )
        order = np.asarray(state.get("sort_order_out"), dtype=np.int64)
        if int(order.size) == 0:
            state["sorted_pdg_id_out"] = np.zeros((0,), dtype=np.int32)
            state["has_pdg_id_out"] = bool(
                chunk_in.has_raw(chunk_in.values_key(self.pdg_id_field))
                and chunk_in.has_raw(chunk_in.offsets_key(self.pdg_id_field, 0))
            )
            return

        has_pdg = chunk_in.has_raw(chunk_in.values_key(self.pdg_id_field)) and chunk_in.has_raw(
            chunk_in.offsets_key(self.pdg_id_field, 0)
        )
        if not has_pdg:
            state["sorted_pdg_id_out"] = np.zeros((int(order.size),), dtype=np.int32)
            state["has_pdg_id_out"] = False
            return

        pdg_offsets = chunk_in.offsets(self.pdg_id_field, 0)
        pdg_counts = (pdg_offsets[1:] - pdg_offsets[:-1]).astype(np.int64, copy=False)
        hit_counts = np.asarray(state["layout"]["hit_counts"], dtype=np.int64)
        if not np.array_equal(pdg_counts, hit_counts):
            raise RuntimeError("hits_pdg_id list lengths do not match event hit counts.")

        state["sorted_pdg_id_out"] = chunk_in.values(self.pdg_id_field)[order].astype(np.int32, copy=False)
        state["has_pdg_id_out"] = True
