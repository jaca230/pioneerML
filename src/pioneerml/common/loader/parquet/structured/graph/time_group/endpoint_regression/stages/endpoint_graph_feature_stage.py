from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import numpy as np

from pioneerml.common.loader.stage.stages import GraphFeatureStage


class EndpointGraphFeatureStage(GraphFeatureStage):
    """Build endpoint graph-level features `[pred_pion, pred_muon, pred_mip, total_group_edep]`."""

    name = "build_graph_features"
    requires = ("layout", "local_gid", "row_ids_graph")
    provides = ("x_graph_out",)

    def __init__(
        self,
        *,
        input_state_key: str = "features_in",
        graph_feature_dim: int = 4,
    ) -> None:
        super().__init__(input_state_key=input_state_key)
        self.graph_feature_dim = int(graph_feature_dim)

    def run(self, *, state: MutableMapping[str, Any], loader) -> None:
        layout = state["layout"]
        total_graphs = int(layout["total_graphs"])
        local_gid = state["local_gid"]
        row_ids_graph = state["row_ids_graph"]
        chunk_in = self.get_input_store(state=state)

        x_graph_out = np.zeros((total_graphs, self.graph_feature_dim), dtype=np.float32)

        has_group_prob_columns = all(
            chunk_in.has_raw(chunk_in.values_key(name)) and chunk_in.has_raw(chunk_in.offsets_key(name, 0))
            for name in ("pred_pion", "pred_muon", "pred_mip")
        )
        if has_group_prob_columns:
            self.fill_graph_column_from_group_values(
                out=x_graph_out,
                dst_col=0,
                vals=chunk_in.values("pred_pion"),
                offs=chunk_in.offsets("pred_pion", 0),
                total_graphs=total_graphs,
                local_gid=local_gid,
                row_ids_graph=row_ids_graph,
            )
            self.fill_graph_column_from_group_values(
                out=x_graph_out,
                dst_col=1,
                vals=chunk_in.values("pred_muon"),
                offs=chunk_in.offsets("pred_muon", 0),
                total_graphs=total_graphs,
                local_gid=local_gid,
                row_ids_graph=row_ids_graph,
            )
            self.fill_graph_column_from_group_values(
                out=x_graph_out,
                dst_col=2,
                vals=chunk_in.values("pred_mip"),
                offs=chunk_in.offsets("pred_mip", 0),
                total_graphs=total_graphs,
                local_gid=local_gid,
                row_ids_graph=row_ids_graph,
            )
        elif bool(getattr(loader, "include_targets", False)):
            group_truth_out = state.get("group_truth_out")
            if isinstance(group_truth_out, np.ndarray) and group_truth_out.shape[0] == total_graphs:
                x_graph_out[:, 0:3] = group_truth_out[:, 0:3]

        # Group-level total deposited energy (legacy "u" feature).
        if total_graphs > 0:
            sum_e = self.graph_weighted_sum_from_nodes(
                global_group_id=layout["global_group_id"],
                values=chunk_in.values("hits_edep"),
                total_graphs=total_graphs,
            )
            x_graph_out[:, 3] = sum_e

        state["x_graph_out"] = x_graph_out
