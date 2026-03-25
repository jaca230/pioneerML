from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import numpy as np

from pioneerml.common.data_loader.loaders.array_store import NDArrayStore
from pioneerml.common.data_loader.loaders.stage.stages import BaseStage


class PionStopGraphFeatureStage(BaseStage):
    """Build graph-level priors for pion-stop regression."""

    name = "build_graph_features"
    requires = ("layout", "local_gid", "row_ids_graph")
    provides = (
        "x_graph_out",
        "group_probs_out",
        "endpoint_preds_out",
        "event_affinity_out",
        "pion_stop_preds_out",
    )

    ENDPOINT_BASE_COLUMNS = (
        "pred_group_start_x",
        "pred_group_start_y",
        "pred_group_start_z",
        "pred_group_end_x",
        "pred_group_end_y",
        "pred_group_end_z",
    )
    ENDPOINT_QUANTILE_SUFFIXES = ("q16", "q50", "q84")
    PION_STOP_PRIOR_COLUMNS = (
        "pred_pion_stop_x_q50",
        "pred_pion_stop_y_q50",
        "pred_pion_stop_z_q50",
    )

    def __init__(
        self,
        *,
        input_state_key: str = "features_in",
        use_group_probs: bool = True,
        use_endpoint_preds: bool = True,
        use_event_splitter_affinity: bool = True,
        use_pion_stop_preds: bool = False,
        num_classes: int = 3,
        endpoint_dim: int = 18,
        event_affinity_dim: int = 3,
        pion_stop_dim: int = 3,
    ) -> None:
        self.input_state_key = str(input_state_key)
        self.use_group_probs = bool(use_group_probs)
        self.use_endpoint_preds = bool(use_endpoint_preds)
        self.use_event_splitter_affinity = bool(use_event_splitter_affinity)
        self.use_pion_stop_preds = bool(use_pion_stop_preds)
        self.num_classes = int(num_classes)
        self.endpoint_dim = int(endpoint_dim)
        self.event_affinity_dim = int(event_affinity_dim)
        self.pion_stop_dim = int(pion_stop_dim)

    @classmethod
    def endpoint_quantile_columns(cls) -> tuple[str, ...]:
        out: list[str] = []
        for base in cls.ENDPOINT_BASE_COLUMNS:
            for suffix in cls.ENDPOINT_QUANTILE_SUFFIXES:
                out.append(f"{base}_{suffix}")
        return tuple(out)

    @staticmethod
    def _fill_graph_column_from_group_values(
        *,
        out: np.ndarray,
        dst_col: int,
        vals: np.ndarray,
        offs: np.ndarray,
        total_graphs: int,
        local_gid: np.ndarray,
        row_ids_graph: np.ndarray,
    ) -> None:
        if total_graphs == 0 or vals.size == 0:
            return
        counts = (offs[1:] - offs[:-1]).astype(np.int64, copy=False)
        valid = local_gid < counts[row_ids_graph]
        if not np.any(valid):
            return
        idx = offs[row_ids_graph[valid]] + local_gid[valid]
        out[valid, dst_col] = vals[idx].astype(np.float32, copy=False)

    @staticmethod
    def _fill_event_affinity_from_lists(
        *,
        event_affinity_out: np.ndarray,
        total_graphs: int,
        n_rows: int,
        row_group_counts: np.ndarray,
        row_group_base: np.ndarray,
        hit_time_group_offsets: np.ndarray,
        hit_time_group_values: np.ndarray,
        edge_src_offsets: np.ndarray,
        edge_src_values: np.ndarray,
        edge_dst_offsets: np.ndarray,
        edge_dst_values: np.ndarray,
        edge_aff_offsets: np.ndarray,
        edge_aff_values: np.ndarray,
    ) -> None:
        if total_graphs <= 0:
            return

        for row in range(n_rows):
            gcount = int(row_group_counts[row])
            if gcount <= 0:
                continue

            h0 = int(hit_time_group_offsets[row])
            h1 = int(hit_time_group_offsets[row + 1])
            if h1 > h0:
                tg_map = hit_time_group_values[h0:h1].astype(np.int64, copy=False)
            else:
                tg_map = np.arange(gcount, dtype=np.int64)

            e0 = int(edge_src_offsets[row])
            e1 = int(edge_src_offsets[row + 1])
            d0 = int(edge_dst_offsets[row])
            d1 = int(edge_dst_offsets[row + 1])
            a0 = int(edge_aff_offsets[row])
            a1 = int(edge_aff_offsets[row + 1])
            if e1 <= e0 or d1 <= d0 or a1 <= a0:
                continue

            src = edge_src_values[e0:e1].astype(np.int64, copy=False)
            dst = edge_dst_values[d0:d1].astype(np.int64, copy=False)
            aff = edge_aff_values[a0:a1].astype(np.float32, copy=False)
            if src.size != dst.size or src.size != aff.size:
                raise RuntimeError("Event-splitter edge arrays have inconsistent lengths.")
            if src.size == 0 or tg_map.size == 0:
                continue

            valid = (src >= 0) & (dst >= 0) & (src < tg_map.size) & (dst < tg_map.size)
            if not np.any(valid):
                continue
            src_gid = tg_map[src[valid]]
            dst_gid = tg_map[dst[valid]]
            aff_vals = aff[valid]
            if aff_vals.size == 0:
                continue

            gids = np.unique(np.concatenate((src_gid, dst_gid)))
            graph_base = int(row_group_base[row])
            denom = float(max(1, 2 * max(gcount - 1, 0)))
            for gid in gids.tolist():
                gid_i = int(gid)
                if gid_i < 0 or gid_i >= gcount:
                    continue
                mask = (src_gid == gid_i) | (dst_gid == gid_i)
                if not np.any(mask):
                    continue
                vals = aff_vals[mask]
                global_gid = graph_base + gid_i
                if global_gid < 0 or global_gid >= total_graphs:
                    continue
                event_affinity_out[global_gid, 0] = float(vals.mean())
                event_affinity_out[global_gid, 1] = float(vals.max())
                event_affinity_out[global_gid, 2] = float(vals.size) / denom

    def run_loader(self, *, state: MutableMapping[str, Any], owner) -> None:
        _ = owner
        layout = state["layout"]
        chunk_in = state.get(self.input_state_key)
        if not isinstance(chunk_in, NDArrayStore):
            raise RuntimeError(
                f"Stage '{self.name}' missing required input state map: {self.input_state_key}"
            )

        total_graphs = int(layout["total_graphs"])
        n_rows = int(state["n_rows"])
        row_group_counts = np.asarray(layout["row_group_counts"], dtype=np.int64)
        row_group_base = np.asarray(layout["row_group_base"], dtype=np.int64)
        local_gid = np.asarray(state["local_gid"], dtype=np.int64)
        row_ids_graph = np.asarray(state["row_ids_graph"], dtype=np.int64)

        group_probs_out = np.zeros((total_graphs, self.num_classes), dtype=np.float32)
        endpoint_preds_out = np.zeros((total_graphs, self.endpoint_dim), dtype=np.float32)
        event_affinity_out = np.zeros((total_graphs, self.event_affinity_dim), dtype=np.float32)
        pion_stop_preds_out = np.zeros((total_graphs, self.pion_stop_dim), dtype=np.float32)

        has_group_probs = all(
            chunk_in.has_raw(chunk_in.values_key(name)) and chunk_in.has_raw(chunk_in.offsets_key(name, 0))
            for name in ("pred_pion", "pred_muon", "pred_mip")
        )
        if self.use_group_probs and has_group_probs:
            pred_counts = (chunk_in.offsets("pred_pion", 0)[1:] - chunk_in.offsets("pred_pion", 0)[:-1]).astype(
                np.int64,
                copy=False,
            )
            if not np.array_equal(pred_counts, row_group_counts):
                raise RuntimeError("pred_pion list lengths do not match inferred row group counts.")

            self._fill_graph_column_from_group_values(
                out=group_probs_out,
                dst_col=0,
                vals=chunk_in.values("pred_pion"),
                offs=chunk_in.offsets("pred_pion", 0),
                total_graphs=total_graphs,
                local_gid=local_gid,
                row_ids_graph=row_ids_graph,
            )
            self._fill_graph_column_from_group_values(
                out=group_probs_out,
                dst_col=1,
                vals=chunk_in.values("pred_muon"),
                offs=chunk_in.offsets("pred_muon", 0),
                total_graphs=total_graphs,
                local_gid=local_gid,
                row_ids_graph=row_ids_graph,
            )
            self._fill_graph_column_from_group_values(
                out=group_probs_out,
                dst_col=2,
                vals=chunk_in.values("pred_mip"),
                offs=chunk_in.offsets("pred_mip", 0),
                total_graphs=total_graphs,
                local_gid=local_gid,
                row_ids_graph=row_ids_graph,
            )
        else:
            fallback = state.get("group_truth_out")
            if isinstance(fallback, np.ndarray) and fallback.shape == group_probs_out.shape:
                group_probs_out[:] = fallback.astype(np.float32, copy=False)

        if self.use_endpoint_preds and total_graphs > 0:
            quant_cols = self.endpoint_quantile_columns()
            has_quantile = all(
                chunk_in.has_raw(chunk_in.values_key(name)) and chunk_in.has_raw(chunk_in.offsets_key(name, 0))
                for name in quant_cols
            )
            has_base = all(
                chunk_in.has_raw(chunk_in.values_key(name)) and chunk_in.has_raw(chunk_in.offsets_key(name, 0))
                for name in self.ENDPOINT_BASE_COLUMNS
            )
            if has_quantile:
                for base_idx, base_name in enumerate(self.ENDPOINT_BASE_COLUMNS):
                    for q_idx, q_suffix in enumerate(self.ENDPOINT_QUANTILE_SUFFIXES):
                        field = f"{base_name}_{q_suffix}"
                        counts = (
                            chunk_in.offsets(field, 0)[1:] - chunk_in.offsets(field, 0)[:-1]
                        ).astype(np.int64, copy=False)
                        if not np.array_equal(counts, row_group_counts):
                            raise RuntimeError(f"{field} list lengths do not match inferred row group counts.")
                        self._fill_graph_column_from_group_values(
                            out=endpoint_preds_out,
                            dst_col=(base_idx * 3) + q_idx,
                            vals=chunk_in.values(field),
                            offs=chunk_in.offsets(field, 0),
                            total_graphs=total_graphs,
                            local_gid=local_gid,
                            row_ids_graph=row_ids_graph,
                        )
            elif has_base:
                for base_idx, base_name in enumerate(self.ENDPOINT_BASE_COLUMNS):
                    counts = (
                        chunk_in.offsets(base_name, 0)[1:] - chunk_in.offsets(base_name, 0)[:-1]
                    ).astype(np.int64, copy=False)
                    if not np.array_equal(counts, row_group_counts):
                        raise RuntimeError(f"{base_name} list lengths do not match inferred row group counts.")
                    tmp = np.zeros((total_graphs, 1), dtype=np.float32)
                    self._fill_graph_column_from_group_values(
                        out=tmp,
                        dst_col=0,
                        vals=chunk_in.values(base_name),
                        offs=chunk_in.offsets(base_name, 0),
                        total_graphs=total_graphs,
                        local_gid=local_gid,
                        row_ids_graph=row_ids_graph,
                    )
                    endpoint_preds_out[:, (base_idx * 3) + 0] = tmp[:, 0]
                    endpoint_preds_out[:, (base_idx * 3) + 1] = tmp[:, 0]
                    endpoint_preds_out[:, (base_idx * 3) + 2] = tmp[:, 0]

        has_event_splitter = all(
            chunk_in.has_raw(chunk_in.values_key(name)) and chunk_in.has_raw(chunk_in.offsets_key(name, 0))
            for name in ("edge_src_index", "edge_dst_index", "pred_edge_affinity")
        )
        if self.use_event_splitter_affinity and has_event_splitter:
            self._fill_event_affinity_from_lists(
                event_affinity_out=event_affinity_out,
                total_graphs=total_graphs,
                n_rows=n_rows,
                row_group_counts=row_group_counts,
                row_group_base=row_group_base,
                hit_time_group_offsets=chunk_in.offsets("hits_time_group", 0),
                hit_time_group_values=chunk_in.values("hits_time_group"),
                edge_src_offsets=chunk_in.offsets("edge_src_index", 0),
                edge_src_values=chunk_in.values("edge_src_index"),
                edge_dst_offsets=chunk_in.offsets("edge_dst_index", 0),
                edge_dst_values=chunk_in.values("edge_dst_index"),
                edge_aff_offsets=chunk_in.offsets("pred_edge_affinity", 0),
                edge_aff_values=chunk_in.values("pred_edge_affinity"),
            )

        has_pion_stop_priors = all(
            chunk_in.has_raw(chunk_in.values_key(name)) and chunk_in.has_raw(chunk_in.offsets_key(name, 0))
            for name in self.PION_STOP_PRIOR_COLUMNS
        )
        if self.use_pion_stop_preds and has_pion_stop_priors:
            for i, field_name in enumerate(self.PION_STOP_PRIOR_COLUMNS):
                counts = (
                    chunk_in.offsets(field_name, 0)[1:] - chunk_in.offsets(field_name, 0)[:-1]
                ).astype(np.int64, copy=False)
                if not np.array_equal(counts, row_group_counts):
                    raise RuntimeError(f"{field_name} list lengths do not match inferred row group counts.")
                self._fill_graph_column_from_group_values(
                    out=pion_stop_preds_out,
                    dst_col=i,
                    vals=chunk_in.values(field_name),
                    offs=chunk_in.offsets(field_name, 0),
                    total_graphs=total_graphs,
                    local_gid=local_gid,
                    row_ids_graph=row_ids_graph,
                )

        state["group_probs_out"] = group_probs_out
        state["endpoint_preds_out"] = endpoint_preds_out
        state["event_affinity_out"] = event_affinity_out
        state["pion_stop_preds_out"] = pion_stop_preds_out
        state["x_graph_out"] = group_probs_out.copy()
