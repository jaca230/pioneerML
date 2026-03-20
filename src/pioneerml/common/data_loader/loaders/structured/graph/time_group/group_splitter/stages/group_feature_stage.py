from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import numpy as np

from pioneerml.common.data_loader.loaders.stage.stages import GraphFeatureStage


class GroupFeatureStage(GraphFeatureStage):
    """Build per-graph features used by GroupSplitter model."""

    name = "build_graph_features"
    requires = ("layout", "local_gid", "row_ids_graph")
    provides = ("x_graph_out",)

    def __init__(
        self,
        *,
        input_state_key: str = "features_in",
        num_classes: int = 3,
    ) -> None:
        super().__init__(input_state_key=input_state_key)
        self.num_classes = int(num_classes)

    def run_loader(self, *, state: MutableMapping[str, Any], owner) -> None:
        layout = state["layout"]
        total_graphs = int(layout["total_graphs"])
        local_gid = state["local_gid"]
        row_ids_graph = state["row_ids_graph"]

        chunk_in = self.get_input_store(state=state)

        x_graph_out = np.zeros((total_graphs, self.num_classes), dtype=np.float32)
        has_prob_columns = all(
            chunk_in.has_raw(chunk_in.values_key(name))
            and chunk_in.has_raw(chunk_in.offsets_key(name, 0))
            for name in ("pred_pion", "pred_muon", "pred_mip")
        )

        if has_prob_columns:
            p_off = chunk_in.offsets("pred_pion", 0)
            m_off = chunk_in.offsets("pred_muon", 0)
            i_off = chunk_in.offsets("pred_mip", 0)
            row_group_counts = layout["row_group_counts"]
            pred_counts = (p_off[1:] - p_off[:-1]).astype(np.int64, copy=False)
            if not np.array_equal(pred_counts, row_group_counts):
                raise RuntimeError("Group-probability list lengths do not match inferred row group counts.")

            self.fill_graph_column_from_group_values(
                out=x_graph_out,
                dst_col=0,
                vals=chunk_in.values("pred_pion"),
                offs=p_off,
                total_graphs=total_graphs,
                local_gid=local_gid,
                row_ids_graph=row_ids_graph,
            )
            self.fill_graph_column_from_group_values(
                out=x_graph_out,
                dst_col=1,
                vals=chunk_in.values("pred_muon"),
                offs=m_off,
                total_graphs=total_graphs,
                local_gid=local_gid,
                row_ids_graph=row_ids_graph,
            )
            self.fill_graph_column_from_group_values(
                out=x_graph_out,
                dst_col=2,
                vals=chunk_in.values("pred_mip"),
                offs=i_off,
                total_graphs=total_graphs,
                local_gid=local_gid,
                row_ids_graph=row_ids_graph,
            )
        elif bool(getattr(owner, "include_targets", False)):
            group_truth = state.get("group_truth_out")
            if isinstance(group_truth, np.ndarray) and group_truth.shape == x_graph_out.shape:
                x_graph_out[:] = group_truth

        state["x_graph_out"] = x_graph_out
