from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import numpy as np
import torch

from pioneerml.common.data_loader.loaders.stage.stages.base_target_stage import BaseTargetStage


class PositronAngleRelevantGraphFilterStage(BaseTargetStage):
    """Optionally keep only positron-like time-groups for training supervision."""

    name = "filter_relevant_graphs"
    requires = (
        "layout",
        "x_out",
        "tgroup_out",
        "coord_sorted",
        "z_sorted",
        "e_sorted",
        "view_sorted",
        "sorted_group_ids_out",
        "sorted_pdg_id_out",
        "has_pdg_id_out",
        "group_probs_out",
        "splitter_probs_out",
        "endpoint_preds_out",
        "event_affinity_out",
        "pion_stop_preds_out",
        "graph_event_id",
        "graph_time_group_id",
        "row_ids_graph",
        "local_gid",
    )
    provides = (
        "layout",
        "x_out",
        "tgroup_out",
        "coord_sorted",
        "z_sorted",
        "e_sorted",
        "view_sorted",
        "sorted_group_ids_out",
        "sorted_pdg_id_out",
        "group_probs_out",
        "splitter_probs_out",
        "endpoint_preds_out",
        "event_affinity_out",
        "pion_stop_preds_out",
        "graph_event_id",
        "graph_time_group_id",
        "row_ids_graph",
        "local_gid",
        "y_graph",
    )

    def __init__(
        self,
        *,
        training_relevant_only: bool = True,
        min_relevant_hits: int = 2,
        positron_pdg_id: int = -11,
    ) -> None:
        self.training_relevant_only = bool(training_relevant_only)
        self.min_relevant_hits = max(1, int(min_relevant_hits))
        self.positron_pdg_id = int(positron_pdg_id)

    @staticmethod
    def _subset_node_array(
        *,
        arr: np.ndarray,
        node_ptr: np.ndarray,
        keep_graph_ids: np.ndarray,
        out_shape_tail: tuple[int, ...],
    ) -> np.ndarray:
        keep = keep_graph_ids.astype(np.int64, copy=False)
        counts = (node_ptr[keep + 1] - node_ptr[keep]).astype(np.int64, copy=False)
        out_ptr = np.zeros((int(keep.size) + 1,), dtype=np.int64)
        out_ptr[1:] = np.cumsum(counts, dtype=np.int64)
        total_nodes = int(out_ptr[-1])
        out = np.empty((total_nodes, *out_shape_tail), dtype=arr.dtype)
        for new_gid, old_gid in enumerate(keep.tolist()):
            old_n0 = int(node_ptr[old_gid])
            old_n1 = int(node_ptr[old_gid + 1])
            new_n0 = int(out_ptr[new_gid])
            new_n1 = int(out_ptr[new_gid + 1])
            out[new_n0:new_n1] = arr[old_n0:old_n1]
        return out

    @staticmethod
    def _empty_graph_array_like(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 1:
            return np.zeros((0,), dtype=arr.dtype)
        return np.zeros((0, *arr.shape[1:]), dtype=arr.dtype)

    def run_loader(self, *, state: MutableMapping[str, Any], owner) -> None:
        if not self.include_targets(owner=owner, state=state):
            return
        if not self.training_relevant_only:
            return

        layout = dict(state["layout"])
        total_graphs = int(layout["total_graphs"])
        if total_graphs <= 0:
            return

        sorted_group_ids = np.asarray(state.get("sorted_group_ids_out"), dtype=np.int64)
        sorted_pdg_id = np.asarray(state.get("sorted_pdg_id_out"), dtype=np.int32)
        has_pdg_id = bool(state.get("has_pdg_id_out", False))
        if not has_pdg_id:
            raise RuntimeError(
                "training_relevant_only=True requires 'hits_pdg_id' in loader inputs for positron-angle training."
            )
        if sorted_group_ids.size != sorted_pdg_id.size:
            raise RuntimeError(
                "sorted_group_ids_out and sorted_pdg_id_out must be aligned for positron filtering."
            )

        is_positron = sorted_pdg_id == int(self.positron_pdg_id)
        if sorted_pdg_id.size == 0 or not np.any(is_positron):
            keep_graph_ids = np.zeros((0,), dtype=np.int64)
        else:
            counts = np.bincount(sorted_group_ids[is_positron], minlength=total_graphs).astype(np.int64, copy=False)
            keep_graph_ids = np.flatnonzero(counts >= int(self.min_relevant_hits)).astype(np.int64, copy=False)

        if int(keep_graph_ids.size) == total_graphs:
            return

        node_ptr = np.asarray(layout["node_ptr"], dtype=np.int64)
        node_counts = np.asarray(layout["node_counts"], dtype=np.int64)
        keep_node_counts = (
            node_counts[keep_graph_ids].astype(np.int64, copy=False)
            if int(keep_graph_ids.size) > 0
            else np.zeros((0,), dtype=np.int64)
        )
        new_node_ptr = np.zeros((int(keep_node_counts.size) + 1,), dtype=np.int64)
        if keep_node_counts.size > 0:
            new_node_ptr[1:] = np.cumsum(keep_node_counts, dtype=np.int64)
        new_total_nodes = int(new_node_ptr[-1])
        new_total_graphs = int(keep_node_counts.size)
        edge_counts = keep_node_counts * np.maximum(keep_node_counts - 1, 0)
        new_edge_ptr = np.zeros((new_total_graphs + 1,), dtype=np.int64)
        if edge_counts.size > 0:
            new_edge_ptr[1:] = np.cumsum(edge_counts, dtype=np.int64)
        new_total_edges = int(new_edge_ptr[-1])

        if new_total_graphs == 0:
            state["layout"] = {
                **layout,
                "total_graphs": 0,
                "total_nodes": 0,
                "total_edges": 0,
                "node_counts": np.zeros((0,), dtype=np.int64),
                "node_ptr": np.zeros((1,), dtype=np.int64),
                "edge_ptr": np.zeros((1,), dtype=np.int64),
            }
            state["x_out"] = np.zeros((0, state["x_out"].shape[1]), dtype=state["x_out"].dtype)
            state["tgroup_out"] = np.zeros((0,), dtype=state["tgroup_out"].dtype)
            state["coord_sorted"] = np.zeros((0,), dtype=state["coord_sorted"].dtype)
            state["z_sorted"] = np.zeros((0,), dtype=state["z_sorted"].dtype)
            state["e_sorted"] = np.zeros((0,), dtype=state["e_sorted"].dtype)
            state["view_sorted"] = np.zeros((0,), dtype=state["view_sorted"].dtype)
            state["sorted_group_ids_out"] = np.zeros((0,), dtype=np.int64)
            state["sorted_pdg_id_out"] = np.zeros((0,), dtype=np.int32)
            state["sort_order_out"] = np.zeros((0,), dtype=np.int64)
            state["group_probs_out"] = self._empty_graph_array_like(np.asarray(state["group_probs_out"]))
            state["endpoint_preds_out"] = self._empty_graph_array_like(np.asarray(state["endpoint_preds_out"]))
            state["event_affinity_out"] = self._empty_graph_array_like(np.asarray(state["event_affinity_out"]))
            state["pion_stop_preds_out"] = self._empty_graph_array_like(np.asarray(state["pion_stop_preds_out"]))
            state["graph_event_id"] = np.zeros((0,), dtype=np.int64)
            state["graph_time_group_id"] = np.zeros((0,), dtype=np.int64)
            state["row_ids_graph"] = np.zeros((0,), dtype=np.int64)
            state["local_gid"] = np.zeros((0,), dtype=np.int64)
            if isinstance(state.get("splitter_probs_out"), np.ndarray):
                splitter = np.asarray(state["splitter_probs_out"])
                state["splitter_probs_out"] = np.zeros((0, splitter.shape[1]), dtype=splitter.dtype)
            if isinstance(state.get("node_truth_out"), np.ndarray):
                node_truth = np.asarray(state["node_truth_out"])
                state["node_truth_out"] = np.zeros((0, node_truth.shape[1]), dtype=node_truth.dtype)
            if isinstance(state.get("group_truth_out"), np.ndarray):
                group_truth = np.asarray(state["group_truth_out"])
                state["group_truth_out"] = np.zeros((0, group_truth.shape[1]), dtype=group_truth.dtype)
            y_graph = state.get("y_graph")
            if isinstance(y_graph, torch.Tensor):
                state["y_graph"] = y_graph[:0]
            return

        x_out = np.asarray(state["x_out"])
        tgroup_out = np.asarray(state["tgroup_out"])
        coord_sorted = np.asarray(state["coord_sorted"])
        z_sorted = np.asarray(state["z_sorted"])
        e_sorted = np.asarray(state["e_sorted"])
        view_sorted = np.asarray(state["view_sorted"])
        splitter_probs_out = np.asarray(state["splitter_probs_out"])
        sorted_pdg_id = np.asarray(state["sorted_pdg_id_out"])
        node_truth_out = np.asarray(state.get("node_truth_out")) if isinstance(state.get("node_truth_out"), np.ndarray) else None

        state["x_out"] = self._subset_node_array(
            arr=x_out,
            node_ptr=node_ptr,
            keep_graph_ids=keep_graph_ids,
            out_shape_tail=(x_out.shape[1],),
        )
        state["tgroup_out"] = self._subset_node_array(
            arr=tgroup_out.reshape(-1, 1),
            node_ptr=node_ptr,
            keep_graph_ids=keep_graph_ids,
            out_shape_tail=(1,),
        ).reshape(-1)
        state["coord_sorted"] = self._subset_node_array(
            arr=coord_sorted.reshape(-1, 1),
            node_ptr=node_ptr,
            keep_graph_ids=keep_graph_ids,
            out_shape_tail=(1,),
        ).reshape(-1)
        state["z_sorted"] = self._subset_node_array(
            arr=z_sorted.reshape(-1, 1),
            node_ptr=node_ptr,
            keep_graph_ids=keep_graph_ids,
            out_shape_tail=(1,),
        ).reshape(-1)
        state["e_sorted"] = self._subset_node_array(
            arr=e_sorted.reshape(-1, 1),
            node_ptr=node_ptr,
            keep_graph_ids=keep_graph_ids,
            out_shape_tail=(1,),
        ).reshape(-1)
        state["view_sorted"] = self._subset_node_array(
            arr=view_sorted.reshape(-1, 1),
            node_ptr=node_ptr,
            keep_graph_ids=keep_graph_ids,
            out_shape_tail=(1,),
        ).reshape(-1)
        state["splitter_probs_out"] = self._subset_node_array(
            arr=splitter_probs_out,
            node_ptr=node_ptr,
            keep_graph_ids=keep_graph_ids,
            out_shape_tail=(splitter_probs_out.shape[1],),
        )
        state["sorted_pdg_id_out"] = self._subset_node_array(
            arr=sorted_pdg_id.reshape(-1, 1),
            node_ptr=node_ptr,
            keep_graph_ids=keep_graph_ids,
            out_shape_tail=(1,),
        ).reshape(-1)
        if node_truth_out is not None:
            state["node_truth_out"] = self._subset_node_array(
                arr=node_truth_out,
                node_ptr=node_ptr,
                keep_graph_ids=keep_graph_ids,
                out_shape_tail=(node_truth_out.shape[1],),
            )
        state["sorted_group_ids_out"] = np.repeat(
            np.arange(new_total_graphs, dtype=np.int64),
            keep_node_counts,
        )
        state["sort_order_out"] = np.arange(new_total_nodes, dtype=np.int64)

        state["group_probs_out"] = np.asarray(state["group_probs_out"])[keep_graph_ids]
        state["endpoint_preds_out"] = np.asarray(state["endpoint_preds_out"])[keep_graph_ids]
        state["event_affinity_out"] = np.asarray(state["event_affinity_out"])[keep_graph_ids]
        state["pion_stop_preds_out"] = np.asarray(state["pion_stop_preds_out"])[keep_graph_ids]
        state["graph_event_id"] = np.asarray(state["graph_event_id"])[keep_graph_ids]
        state["graph_time_group_id"] = np.asarray(state["graph_time_group_id"])[keep_graph_ids]
        state["row_ids_graph"] = np.asarray(state["row_ids_graph"])[keep_graph_ids]
        state["local_gid"] = np.asarray(state["local_gid"])[keep_graph_ids]
        if isinstance(state.get("group_truth_out"), np.ndarray):
            state["group_truth_out"] = np.asarray(state["group_truth_out"])[keep_graph_ids]
        y_graph = state.get("y_graph")
        if isinstance(y_graph, torch.Tensor):
            idx = torch.from_numpy(keep_graph_ids.astype(np.int64, copy=False))
            state["y_graph"] = y_graph.index_select(0, idx)

        state["layout"] = {
            **layout,
            "total_graphs": new_total_graphs,
            "total_nodes": new_total_nodes,
            "total_edges": new_total_edges,
            "node_counts": keep_node_counts,
            "node_ptr": new_node_ptr,
            "edge_ptr": new_edge_ptr,
        }
