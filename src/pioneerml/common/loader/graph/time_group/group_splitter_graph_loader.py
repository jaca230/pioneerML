from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import get_worker_info
from torch_geometric.data import Data

from pioneerml.common.parquet import ParquetChunkReader

from .time_group_graph_loader import TimeGroupGraphLoader


class GroupSplitterGraphLoader(TimeGroupGraphLoader):
    """Chunked graph loader for group-splitter training and inference."""

    GRAPH_COLUMNS = [
        "event_id",
        "hits_time_group",
        "hits_strip_type",
        "hits_coord",
        "hits_z",
        "hits_edep",
    ]
    TARGET_COLUMNS = ["hits_particle_mask"]
    GROUP_PROB_COLUMNS = ["pred_pion", "pred_muon", "pred_mip"]
    NODE_FEATURE_DIM = 4
    EDGE_FEATURE_DIM = 4
    NUM_CLASSES = 3
    MODE_TRAIN = "train"
    MODE_INFERENCE = "inference"

    def __init__(
        self,
        parquet_paths: list[str],
        *,
        group_probs_parquet_paths: list[str] | None = None,
        mode: str = MODE_TRAIN,
        use_group_probs: bool = True,
        batch_size: int = 64,
        row_groups_per_chunk: int = 4,
        num_workers: int = 0,
        columns: list[str] | None = None,
        split: str | None = None,
        train_fraction: float = 0.9,
        val_fraction: float = 0.05,
        test_fraction: float = 0.05,
        split_seed: int = 0,
        sample_fraction: float | None = None,
    ) -> None:
        mode_norm = str(mode).strip().lower()
        if mode_norm not in (self.MODE_TRAIN, self.MODE_INFERENCE):
            raise ValueError(f"Unsupported mode: {mode}. Expected 'train' or 'inference'.")
        self.mode = mode_norm
        self.use_group_probs = bool(use_group_probs)

        probs_paths: list[str] | None = None
        if group_probs_parquet_paths is not None:
            probs_paths = [str(Path(p).expanduser().resolve()) for p in group_probs_parquet_paths]
            if len(probs_paths) != len(parquet_paths):
                raise ValueError(
                    "group_probs_parquet_paths must match parquet_paths length. "
                    f"Got {len(probs_paths)} vs {len(parquet_paths)}."
                )
            missing = [p for p in probs_paths if not Path(p).exists()]
            if missing:
                raise RuntimeError(f"Missing group_probs parquet path(s): {missing}")
        self.group_probs_parquet_paths = probs_paths

        required = self._required_columns_for_mode(self.mode)
        resolved_columns = list(columns) if columns is not None else self._resolve_default_columns(required)
        super().__init__(
            parquet_paths=parquet_paths,
            batch_size=batch_size,
            row_groups_per_chunk=row_groups_per_chunk,
            num_workers=num_workers,
            columns=resolved_columns,
            split=split,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            split_seed=split_seed,
            sample_fraction=sample_fraction,
        )

        missing = [c for c in required if c not in self.columns]
        if missing:
            raise ValueError(f"Missing required columns for mode={self.mode}: {missing}")

    @staticmethod
    def _required_columns_for_mode(mode: str) -> list[str]:
        if mode == GroupSplitterGraphLoader.MODE_TRAIN:
            return [*GroupSplitterGraphLoader.GRAPH_COLUMNS, *GroupSplitterGraphLoader.TARGET_COLUMNS]
        return list(GroupSplitterGraphLoader.GRAPH_COLUMNS)

    def _resolve_default_columns(self, required: list[str]) -> list[str]:
        out = list(required)
        if self.group_probs_parquet_paths is not None:
            return out
        if self.use_group_probs and self._columns_present_in_all_files(self.GROUP_PROB_COLUMNS):
            out.extend(self.GROUP_PROB_COLUMNS)
        return out

    def _columns_present_in_all_files(self, names: list[str]) -> bool:
        if not self.parquet_paths:
            return False
        for path in self.parquet_paths:
            schema_names = set(pq.read_schema(path).names)
            for name in names:
                if name not in schema_names:
                    return False
        return True

    def _constructor_kwargs(self) -> dict[str, object]:
        kwargs = super()._constructor_kwargs()
        kwargs["group_probs_parquet_paths"] = self.group_probs_parquet_paths
        kwargs["mode"] = self.mode
        kwargs["use_group_probs"] = self.use_group_probs
        return kwargs

    @property
    def include_targets(self) -> bool:
        return self.mode == self.MODE_TRAIN

    def empty_data(self) -> tuple[Data, torch.Tensor]:
        data = Data(
            x=torch.empty((0, self.NODE_FEATURE_DIM), dtype=torch.float32),
            edge_index=torch.empty((2, 0), dtype=torch.int64),
            edge_attr=torch.empty((0, self.EDGE_FEATURE_DIM), dtype=torch.float32),
            group_total_energy=torch.empty((0, 1), dtype=torch.float32),
            group_probs=torch.empty((0, self.NUM_CLASSES), dtype=torch.float32),
            time_group_ids=torch.empty((0,), dtype=torch.int64),
        )
        targets = torch.empty((0, self.NUM_CLASSES), dtype=torch.float32)
        data.num_graphs = 0
        data.num_groups = 0
        data.num_rows = 0
        if self.include_targets:
            data.y = targets
        return data, targets

    def _iter_batches(self, *, shuffle_batches: bool) -> Iterator[Data]:
        row_offset = 0
        for table in self._iter_chunk_tables():
            raw_rows = int(table.num_rows)
            table = self._filter_rows_before_graph_build(table)
            if table is None:
                row_offset += raw_rows
                continue
            chunk = self._build_chunk_graph_arrays(table=table)
            if "graph_event_ids" in chunk and row_offset != 0:
                chunk["graph_event_ids"] = chunk["graph_event_ids"] + int(row_offset)
            num_graphs = int(chunk["num_graphs"])
            if num_graphs <= 0:
                row_offset += raw_rows
                continue

            starts = torch.arange(0, num_graphs, self.batch_size, dtype=torch.int64)
            if shuffle_batches and starts.numel() > 1:
                starts = starts[torch.randperm(starts.numel())]
            for g0 in starts.tolist():
                g1 = min(g0 + self.batch_size, num_graphs)
                yield self._slice_chunk_batch(chunk, g0, g1)
            row_offset += raw_rows

    def _iter_chunk_tables(self) -> Iterator[pa.Table]:
        if self.group_probs_parquet_paths is None:
            reader = ParquetChunkReader(
                parquet_paths=self.parquet_paths,
                columns=self.columns,
                row_groups_per_chunk=self.row_groups_per_chunk,
            )
            yield from reader.iter_tables()
            return

        tasks = self._shard_tasks(self._aligned_row_group_tasks())
        chunk_span = max(1, int(self.row_groups_per_chunk))
        for i in range(0, len(tasks), chunk_span):
            chunk_tasks = tasks[i : i + chunk_span]
            if not chunk_tasks:
                continue
            main_tables: list[pa.Table] = []
            prob_tables: list[pa.Table] = []
            for main_path, prob_path, rg in chunk_tasks:
                main_pf = pq.ParquetFile(main_path)
                prob_pf = pq.ParquetFile(prob_path)
                main_tables.append(main_pf.read_row_group(rg, columns=self.columns))
                prob_tables.append(prob_pf.read_row_group(rg, columns=self.GROUP_PROB_COLUMNS))
            main_table = main_tables[0] if len(main_tables) == 1 else pa.concat_tables(main_tables, promote_options="default")
            prob_table = prob_tables[0] if len(prob_tables) == 1 else pa.concat_tables(prob_tables, promote_options="default")
            if main_table.num_rows != prob_table.num_rows:
                raise RuntimeError(
                    "Aligned chunk row mismatch between main and group_probs tables: "
                    f"{main_table.num_rows} vs {prob_table.num_rows}"
                )
            merged = main_table
            for col in self.GROUP_PROB_COLUMNS:
                if col in merged.column_names:
                    merged = merged.set_column(merged.schema.get_field_index(col), col, prob_table.column(col))
                else:
                    merged = merged.append_column(col, prob_table.column(col))
            yield merged.combine_chunks()

    @staticmethod
    def _shard_tasks(tasks: list[tuple[str, str, int]]) -> list[tuple[str, str, int]]:
        worker = get_worker_info()
        if worker is None:
            return tasks
        return tasks[worker.id :: worker.num_workers]

    def _aligned_row_group_tasks(self) -> list[tuple[str, str, int]]:
        tasks: list[tuple[str, str, int]] = []
        assert self.group_probs_parquet_paths is not None
        for main_path, prob_path in zip(self.parquet_paths, self.group_probs_parquet_paths, strict=True):
            main_pf = pq.ParquetFile(main_path)
            prob_pf = pq.ParquetFile(prob_path)
            if main_pf.num_row_groups != prob_pf.num_row_groups:
                raise RuntimeError(
                    "Row-group mismatch between main and group_probs files: "
                    f"{main_path} ({main_pf.num_row_groups}) vs {prob_path} ({prob_pf.num_row_groups})"
                )
            for rg in range(main_pf.num_row_groups):
                n_main = int(main_pf.metadata.row_group(rg).num_rows)
                n_prob = int(prob_pf.metadata.row_group(rg).num_rows)
                if n_main != n_prob:
                    raise RuntimeError(
                        f"Row count mismatch at row-group {rg}: {main_path} ({n_main}) vs {prob_path} ({n_prob})"
                    )
                tasks.append((main_path, prob_path, rg))
        return tasks

    def _extract_chunk_inputs(self, table: pa.Table) -> dict[str, np.ndarray]:
        hits_time_group_offsets, hits_time_group_values = self._extract_list_column(table, "hits_time_group", np.int64)
        _, hits_coord_values = self._extract_list_column(table, "hits_coord", np.float32)
        _, hits_z_values = self._extract_list_column(table, "hits_z", np.float32)
        _, hits_edep_values = self._extract_list_column(table, "hits_edep", np.float32)
        _, hits_strip_type_values = self._extract_list_column(table, "hits_strip_type", np.int32)

        out: dict[str, np.ndarray] = {
            "event_ids": self._to_np(table.column("event_id").chunk(0), np.int64),
            "hits_time_group_offsets": hits_time_group_offsets,
            "hits_time_group_values": hits_time_group_values,
            "hits_coord_values": hits_coord_values,
            "hits_z_values": hits_z_values,
            "hits_edep_values": hits_edep_values,
            "hits_strip_type_values": hits_strip_type_values,
        }

        if self.include_targets:
            hits_particle_mask_offsets, hits_particle_mask_values = self._extract_list_column(
                table, "hits_particle_mask", np.int32
            )
            out["hits_particle_mask_offsets"] = hits_particle_mask_offsets
            out["hits_particle_mask_values"] = hits_particle_mask_values

        has_probs = all(col in table.column_names for col in self.GROUP_PROB_COLUMNS)
        out["has_group_prob_columns"] = np.asarray([1 if has_probs else 0], dtype=np.int8)
        if has_probs:
            p_off, p_vals = self._extract_list_column(table, "pred_pion", np.float32)
            m_off, m_vals = self._extract_list_column(table, "pred_muon", np.float32)
            i_off, i_vals = self._extract_list_column(table, "pred_mip", np.float32)
            out.update(
                {
                    "pred_pion_offsets": p_off,
                    "pred_pion_values": p_vals,
                    "pred_muon_offsets": m_off,
                    "pred_muon_values": m_vals,
                    "pred_mip_offsets": i_off,
                    "pred_mip_values": i_vals,
                }
            )
        return out

    @staticmethod
    def _particle_mask_to_multihot(mask_values: np.ndarray) -> np.ndarray:
        # Bit layout: 1=pion, 2=muon, 4=mip.
        out = np.zeros((mask_values.size, 3), dtype=np.float32)
        out[:, 0] = ((mask_values & 1) != 0).astype(np.float32, copy=False)
        out[:, 1] = ((mask_values & 2) != 0).astype(np.float32, copy=False)
        out[:, 2] = ((mask_values & 4) != 0).astype(np.float32, copy=False)
        return out

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

    @classmethod
    def _fill_group_probs_from_lists(
        cls,
        *,
        group_probs_out: np.ndarray,
        total_graphs: int,
        local_gid: np.ndarray,
        row_ids_graph: np.ndarray,
        pred_pion_values: np.ndarray,
        pred_pion_offsets: np.ndarray,
        pred_muon_values: np.ndarray,
        pred_muon_offsets: np.ndarray,
        pred_mip_values: np.ndarray,
        pred_mip_offsets: np.ndarray,
    ) -> None:
        cls._fill_graph_column_from_group_values(
            out=group_probs_out,
            dst_col=0,
            vals=pred_pion_values,
            offs=pred_pion_offsets,
            total_graphs=total_graphs,
            local_gid=local_gid,
            row_ids_graph=row_ids_graph,
        )
        cls._fill_graph_column_from_group_values(
            out=group_probs_out,
            dst_col=1,
            vals=pred_muon_values,
            offs=pred_muon_offsets,
            total_graphs=total_graphs,
            local_gid=local_gid,
            row_ids_graph=row_ids_graph,
        )
        cls._fill_graph_column_from_group_values(
            out=group_probs_out,
            dst_col=2,
            vals=pred_mip_values,
            offs=pred_mip_offsets,
            total_graphs=total_graphs,
            local_gid=local_gid,
            row_ids_graph=row_ids_graph,
        )

    @staticmethod
    def _populate_node_tensors(
        *,
        total_nodes: int,
        global_group_id: np.ndarray,
        hit_coord_values: np.ndarray,
        hit_z_values: np.ndarray,
        hit_edep_values: np.ndarray,
        hit_strip_type_values: np.ndarray,
        hit_time_group_values: np.ndarray,
        x_out: np.ndarray,
        time_group_ids_out: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if total_nodes > 0:
            order = np.argsort(global_group_id, kind="stable")
            x_out[:, 0] = hit_coord_values[order]
            x_out[:, 1] = hit_z_values[order]
            x_out[:, 2] = hit_edep_values[order]
            x_out[:, 3] = hit_strip_type_values[order].astype(np.float32, copy=False)
            time_group_ids_out[:] = hit_time_group_values[order]
            return (
                x_out[:, 0],
                x_out[:, 1],
                x_out[:, 2],
                hit_strip_type_values[order],
                order,
            )
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int64),
        )

    def _populate_graph_edges(
        self,
        *,
        node_counts: np.ndarray,
        node_ptr: np.ndarray,
        edge_ptr: np.ndarray,
        sorted_coord_values: np.ndarray,
        sorted_z_values: np.ndarray,
        sorted_edep_values: np.ndarray,
        sorted_strip_type_values: np.ndarray,
        edge_index_out: np.ndarray,
        edge_attr_out: np.ndarray,
    ) -> None:
        edge_tpl_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for k in np.unique(node_counts):
            k = int(k)
            if k <= 1:
                continue
            src, dst = self._complete_digraph_cached(k, edge_tpl_cache)
            ecount = int(src.shape[0])
            graphs = np.flatnonzero(node_counts == k)
            if graphs.size == 0:
                continue
            rel_edge = np.arange(ecount, dtype=np.int64)
            graph_block = int(self.edge_populate_graph_block)
            for i in range(0, graphs.size, graph_block):
                g = graphs[i : i + graph_block]
                node_base = node_ptr[g]
                edge_base = edge_ptr[g]
                pos = (edge_base[:, None] + rel_edge[None, :]).reshape(-1)
                src_idx = (node_base[:, None] + src[None, :]).reshape(-1)
                dst_idx = (node_base[:, None] + dst[None, :]).reshape(-1)
                edge_index_out[0, pos] = src_idx
                edge_index_out[1, pos] = dst_idx
                edge_attr_out[pos, 0] = sorted_coord_values[dst_idx] - sorted_coord_values[src_idx]
                edge_attr_out[pos, 1] = sorted_z_values[dst_idx] - sorted_z_values[src_idx]
                edge_attr_out[pos, 2] = sorted_edep_values[dst_idx] - sorted_edep_values[src_idx]
                edge_attr_out[pos, 3] = (sorted_strip_type_values[src_idx] == sorted_strip_type_values[dst_idx]).astype(
                    np.float32
                )

    def _build_chunk_graph_arrays(self, *, table: pa.Table) -> dict:
        n_rows = int(table.num_rows)
        chunk_in = self._extract_chunk_inputs(table)
        layout = self._compute_group_layout(
            n_rows=n_rows,
            hits_time_group_offsets=chunk_in["hits_time_group_offsets"],
            hits_time_group_values=chunk_in["hits_time_group_values"],
            row_group_count_candidates=[],
        )
        if self.include_targets:
            target_counts = self._counts_from_offsets(chunk_in["hits_particle_mask_offsets"])
            if not np.array_equal(target_counts, layout["hit_counts"]):
                raise RuntimeError("hits_particle_mask length mismatch with hit arrays in chunk.")

        total_nodes = int(layout["total_nodes"])
        total_edges = int(layout["total_edges"])
        total_graphs = int(layout["total_graphs"])
        node_ptr = layout["node_ptr"]
        edge_ptr = layout["edge_ptr"]
        node_counts = layout["node_counts"]
        global_group_id = layout["global_group_id"]
        row_group_counts = layout["row_group_counts"]
        row_group_base = layout["row_group_base"]

        x_out = np.empty((total_nodes, self.NODE_FEATURE_DIM), dtype=np.float32)
        edge_index_out = np.empty((2, total_edges), dtype=np.int64)
        edge_attr_out = np.empty((total_edges, self.EDGE_FEATURE_DIM), dtype=np.float32)
        tgroup_out = np.empty((total_nodes,), dtype=np.int64)
        row_ids_graph, local_gid, graph_event_ids, graph_group_ids = self._build_graph_index_mapping(
            total_graphs=total_graphs,
            n_rows=n_rows,
            row_group_counts=row_group_counts,
            row_group_base=row_group_base,
            event_ids=chunk_in["event_ids"],
        )

        coord_sorted, z_sorted, e_sorted, view_sorted, order = self._populate_node_tensors(
            total_nodes=total_nodes,
            global_group_id=global_group_id,
            hit_coord_values=chunk_in["hits_coord_values"],
            hit_z_values=chunk_in["hits_z_values"],
            hit_edep_values=chunk_in["hits_edep_values"],
            hit_strip_type_values=chunk_in["hits_strip_type_values"],
            hit_time_group_values=chunk_in["hits_time_group_values"],
            x_out=x_out,
            time_group_ids_out=tgroup_out,
        )

        y_node_out = np.zeros((total_nodes, self.NUM_CLASSES), dtype=np.float32)
        group_truth = np.zeros((total_graphs, self.NUM_CLASSES), dtype=np.float32)
        if self.include_targets and total_nodes > 0:
            mask_sorted = chunk_in["hits_particle_mask_values"][order]
            y_node_out[:, :] = self._particle_mask_to_multihot(mask_sorted)
            sorted_gid = global_group_id[order]
            for cls_id in range(self.NUM_CLASSES):
                mask = y_node_out[:, cls_id] > 0.5
                if not np.any(mask):
                    continue
                present = np.bincount(sorted_gid[mask], minlength=total_graphs) > 0
                group_truth[:, cls_id] = present.astype(np.float32, copy=False)

        group_total_energy_out = np.zeros((total_graphs, 1), dtype=np.float32)
        if total_graphs > 0:
            sum_e = np.bincount(
                global_group_id,
                weights=chunk_in["hits_edep_values"].astype(np.float64),
                minlength=total_graphs,
            ).astype(np.float32)
            group_total_energy_out[:, 0] = sum_e

        group_probs_out = np.zeros((total_graphs, self.NUM_CLASSES), dtype=np.float32)
        has_prob_columns = int(chunk_in["has_group_prob_columns"][0]) == 1
        if self.use_group_probs:
            if has_prob_columns:
                pred_counts = (chunk_in["pred_pion_offsets"][1:] - chunk_in["pred_pion_offsets"][:-1]).astype(
                    np.int64, copy=False
                )
                if not np.array_equal(pred_counts, row_group_counts):
                    raise RuntimeError("Group-probability list lengths do not match inferred row group counts.")
                self._fill_group_probs_from_lists(
                    group_probs_out=group_probs_out,
                    total_graphs=total_graphs,
                    local_gid=local_gid,
                    row_ids_graph=row_ids_graph,
                    pred_pion_values=chunk_in["pred_pion_values"],
                    pred_pion_offsets=chunk_in["pred_pion_offsets"],
                    pred_muon_values=chunk_in["pred_muon_values"],
                    pred_muon_offsets=chunk_in["pred_muon_offsets"],
                    pred_mip_values=chunk_in["pred_mip_values"],
                    pred_mip_offsets=chunk_in["pred_mip_offsets"],
                )
            elif self.include_targets:
                group_probs_out[:] = group_truth

        self._populate_graph_edges(
            node_counts=node_counts,
            node_ptr=node_ptr,
            edge_ptr=edge_ptr,
            sorted_coord_values=coord_sorted,
            sorted_z_values=z_sorted,
            sorted_edep_values=e_sorted,
            sorted_strip_type_values=view_sorted,
            edge_index_out=edge_index_out,
            edge_attr_out=edge_attr_out,
        )

        chunk_out = {
            "x": torch.from_numpy(x_out),
            "edge_index": torch.from_numpy(edge_index_out),
            "edge_attr": torch.from_numpy(edge_attr_out),
            "group_total_energy": torch.from_numpy(group_total_energy_out),
            "group_probs": torch.from_numpy(group_probs_out),
            "time_group_ids": torch.from_numpy(tgroup_out),
            "graph_event_ids": torch.from_numpy(graph_event_ids),
            "graph_group_ids": torch.from_numpy(graph_group_ids),
            "node_ptr": torch.from_numpy(node_ptr),
            "edge_ptr": torch.from_numpy(edge_ptr),
            "num_rows": n_rows,
            "num_graphs": int(total_graphs),
        }
        if self.include_targets:
            chunk_out["node_targets"] = torch.from_numpy(y_node_out)
        return chunk_out

    @staticmethod
    def _slice_chunk_batch(chunk: dict, g0: int, g1: int) -> Data:
        node_ptr = chunk["node_ptr"]
        edge_ptr = chunk["edge_ptr"]
        n0 = int(node_ptr[g0].item())
        n1 = int(node_ptr[g1].item())
        e0 = int(edge_ptr[g0].item())
        e1 = int(edge_ptr[g1].item())

        d = Data(
            x=chunk["x"][n0:n1],
            edge_index=(chunk["edge_index"][:, e0:e1] - n0),
            edge_attr=chunk["edge_attr"][e0:e1],
            group_total_energy=chunk["group_total_energy"][g0:g1],
            group_probs=chunk["group_probs"][g0:g1],
            time_group_ids=chunk["time_group_ids"][n0:n1],
        )
        if "node_targets" in chunk:
            d.y = chunk["node_targets"][n0:n1]
        local_counts = (node_ptr[g0 + 1 : g1 + 1] - node_ptr[g0:g1]).to(dtype=torch.int64)
        d.batch = torch.repeat_interleave(torch.arange(g1 - g0, dtype=torch.int64), local_counts)
        d.event_ids = chunk["graph_event_ids"][g0:g1]
        d.group_ids = chunk["graph_group_ids"][g0:g1]
        d.num_graphs = int(g1 - g0)
        d.num_groups = int(g1 - g0)
        return d
