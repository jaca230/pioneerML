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

from ..base_graph_loader import BaseGraphLoader


class EventSplitterGraphLoader(BaseGraphLoader):
    """Chunked event-level graph loader for event-splitter training and inference."""

    GRAPH_COLUMNS = [
        "event_id",
        "hits_time_group",
        "hits_strip_type",
        "hits_coord",
        "hits_z",
        "hits_edep",
    ]
    TARGET_COLUMNS = ["hits_contrib_mc_event_id"]

    GROUP_PROB_COLUMNS = ["pred_pion", "pred_muon", "pred_mip"]
    SPLITTER_PROB_COLUMNS = ["pred_hit_pion", "pred_hit_muon", "pred_hit_mip", "time_group_ids"]

    ENDPOINT_BASE_COLUMNS = [
        "pred_group_start_x",
        "pred_group_start_y",
        "pred_group_start_z",
        "pred_group_end_x",
        "pred_group_end_y",
        "pred_group_end_z",
    ]
    ENDPOINT_QUANTILE_SUFFIXES = ("q16", "q50", "q84")

    NODE_FEATURE_DIM = 4
    EDGE_FEATURE_DIM = 5
    NUM_CLASSES = 3
    ENDPOINT_DIM = 18

    MODE_TRAIN = "train"
    MODE_INFERENCE = "inference"

    def __init__(
        self,
        parquet_paths: list[str],
        *,
        group_probs_parquet_paths: list[str] | None = None,
        group_splitter_parquet_paths: list[str] | None = None,
        endpoint_parquet_paths: list[str] | None = None,
        mode: str = MODE_TRAIN,
        use_group_probs: bool = True,
        use_splitter_probs: bool = True,
        use_endpoint_preds: bool = True,
        batch_size: int = 8,
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
        self._parquet_paths_init = [str(Path(p).expanduser().resolve()) for p in parquet_paths]

        mode_norm = str(mode).strip().lower()
        if mode_norm not in (self.MODE_TRAIN, self.MODE_INFERENCE):
            raise ValueError(f"Unsupported mode: {mode}. Expected 'train' or 'inference'.")
        self.mode = mode_norm

        self.use_group_probs = bool(use_group_probs)
        self.use_splitter_probs = bool(use_splitter_probs)
        self.use_endpoint_preds = bool(use_endpoint_preds)

        self.group_probs_parquet_paths = self._resolve_optional_paths(
            parquet_paths=parquet_paths,
            secondary_paths=group_probs_parquet_paths,
            field_name="group_probs_parquet_paths",
        )
        self.group_splitter_parquet_paths = self._resolve_optional_paths(
            parquet_paths=parquet_paths,
            secondary_paths=group_splitter_parquet_paths,
            field_name="group_splitter_parquet_paths",
        )
        self.endpoint_parquet_paths = self._resolve_optional_paths(
            parquet_paths=parquet_paths,
            secondary_paths=endpoint_parquet_paths,
            field_name="endpoint_parquet_paths",
        )

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
    def _resolve_optional_paths(
        *,
        parquet_paths: list[str],
        secondary_paths: list[str] | None,
        field_name: str,
    ) -> list[str] | None:
        if secondary_paths is None:
            return None
        resolved = [str(Path(p).expanduser().resolve()) for p in secondary_paths]
        if len(resolved) != len(parquet_paths):
            raise ValueError(
                f"{field_name} must match parquet_paths length. "
                f"Got {len(resolved)} vs {len(parquet_paths)}."
            )
        missing = [p for p in resolved if not Path(p).exists()]
        if missing:
            raise RuntimeError(f"Missing {field_name}: {missing}")
        return resolved

    @classmethod
    def _endpoint_quantile_columns(cls) -> list[str]:
        out: list[str] = []
        for base in cls.ENDPOINT_BASE_COLUMNS:
            for q in cls.ENDPOINT_QUANTILE_SUFFIXES:
                out.append(f"{base}_{q}")
        return out

    @staticmethod
    def _required_columns_for_mode(mode: str) -> list[str]:
        if mode == EventSplitterGraphLoader.MODE_TRAIN:
            return [*EventSplitterGraphLoader.GRAPH_COLUMNS, *EventSplitterGraphLoader.TARGET_COLUMNS]
        return list(EventSplitterGraphLoader.GRAPH_COLUMNS)

    def _resolve_default_columns(self, required: list[str]) -> list[str]:
        out = list(required)
        if self.group_probs_parquet_paths is None and self.use_group_probs:
            if self._columns_present_in_all_files(self.GROUP_PROB_COLUMNS):
                out.extend(self.GROUP_PROB_COLUMNS)

        if self.group_splitter_parquet_paths is None and self.use_splitter_probs:
            if self._columns_present_in_all_files(self.SPLITTER_PROB_COLUMNS[:3]):
                out.extend(self.SPLITTER_PROB_COLUMNS[:3])
                if self._columns_present_in_all_files(["time_group_ids"]):
                    out.append("time_group_ids")

        if self.endpoint_parquet_paths is None and self.use_endpoint_preds:
            quant_cols = self._endpoint_quantile_columns()
            if self._columns_present_in_all_files(quant_cols):
                out.extend(quant_cols)
            elif self._columns_present_in_all_files(self.ENDPOINT_BASE_COLUMNS):
                out.extend(self.ENDPOINT_BASE_COLUMNS)
            if self._columns_present_in_all_files(["time_group_ids"]):
                out.append("time_group_ids")

        return out

    def _columns_present_in_all_files(self, names: list[str]) -> bool:
        paths = list(getattr(self, "parquet_paths", []) or self._parquet_paths_init)
        if not paths:
            return False
        for path in paths:
            schema_names = set(pq.read_schema(path).names)
            for name in names:
                if name not in schema_names:
                    return False
        return True

    def _constructor_kwargs(self) -> dict[str, object]:
        kwargs = super()._constructor_kwargs()
        kwargs["group_probs_parquet_paths"] = self.group_probs_parquet_paths
        kwargs["group_splitter_parquet_paths"] = self.group_splitter_parquet_paths
        kwargs["endpoint_parquet_paths"] = self.endpoint_parquet_paths
        kwargs["mode"] = self.mode
        kwargs["use_group_probs"] = self.use_group_probs
        kwargs["use_splitter_probs"] = self.use_splitter_probs
        kwargs["use_endpoint_preds"] = self.use_endpoint_preds
        return kwargs

    @property
    def include_targets(self) -> bool:
        return self.mode == self.MODE_TRAIN

    def empty_data(self) -> tuple[Data, torch.Tensor]:
        data = Data(
            x=torch.empty((0, self.NODE_FEATURE_DIM), dtype=torch.float32),
            edge_index=torch.empty((2, 0), dtype=torch.int64),
            edge_attr=torch.empty((0, self.EDGE_FEATURE_DIM), dtype=torch.float32),
            group_ptr=torch.empty((0,), dtype=torch.int64),
            time_group_ids=torch.empty((0,), dtype=torch.int64),
            group_probs=torch.empty((0, self.NUM_CLASSES), dtype=torch.float32),
            splitter_probs=torch.empty((0, self.NUM_CLASSES), dtype=torch.float32),
            endpoint_preds=torch.empty((0, self.ENDPOINT_DIM), dtype=torch.float32),
        )
        targets = torch.empty((0, 1), dtype=torch.float32)
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
        if (
            self.group_probs_parquet_paths is None
            and self.group_splitter_parquet_paths is None
            and self.endpoint_parquet_paths is None
        ):
            reader = ParquetChunkReader(
                parquet_paths=self.parquet_paths,
                columns=self.columns,
                row_groups_per_chunk=self.row_groups_per_chunk,
            )
            yield from reader.iter_tables()
            return

        tasks = self._shard_tasks(self._aligned_row_group_tasks())
        chunk_span = max(1, int(self.row_groups_per_chunk))
        endpoint_cols = self.ENDPOINT_BASE_COLUMNS + self._endpoint_quantile_columns() + ["time_group_ids"]

        for i in range(0, len(tasks), chunk_span):
            chunk_tasks = tasks[i : i + chunk_span]
            if not chunk_tasks:
                continue

            main_tables: list[pa.Table] = []
            group_prob_tables: list[pa.Table] = []
            splitter_prob_tables: list[pa.Table] = []
            endpoint_tables: list[pa.Table] = []

            for main_path, group_prob_path, splitter_prob_path, endpoint_path, rg in chunk_tasks:
                main_pf = pq.ParquetFile(main_path)
                main_tables.append(main_pf.read_row_group(rg, columns=self.columns))
                if group_prob_path is not None:
                    gp_pf = pq.ParquetFile(group_prob_path)
                    group_prob_tables.append(gp_pf.read_row_group(rg, columns=self.GROUP_PROB_COLUMNS))
                if splitter_prob_path is not None:
                    sp_pf = pq.ParquetFile(splitter_prob_path)
                    splitter_raw = sp_pf.read_row_group(rg, columns=self.SPLITTER_PROB_COLUMNS)
                    splitter_prob_tables.append(
                        self._rename_columns(splitter_raw, {"time_group_ids": "splitter_time_group_ids"})
                    )
                if endpoint_path is not None:
                    ep_pf = pq.ParquetFile(endpoint_path)
                    endpoint_raw = ep_pf.read_row_group(rg, columns=endpoint_cols)
                    endpoint_tables.append(
                        self._rename_columns(endpoint_raw, {"time_group_ids": "endpoint_time_group_ids"})
                    )

            main_table = (
                main_tables[0]
                if len(main_tables) == 1
                else pa.concat_tables(main_tables, promote_options="default")
            )
            merged = main_table

            if group_prob_tables:
                gp_table = (
                    group_prob_tables[0]
                    if len(group_prob_tables) == 1
                    else pa.concat_tables(group_prob_tables, promote_options="default")
                )
                if merged.num_rows != gp_table.num_rows:
                    raise RuntimeError(
                        "Aligned chunk row mismatch between main and group_probs tables: "
                        f"{merged.num_rows} vs {gp_table.num_rows}"
                    )
                merged = self._merge_columns_from_table(merged, gp_table, list(gp_table.column_names))

            if splitter_prob_tables:
                sp_table = (
                    splitter_prob_tables[0]
                    if len(splitter_prob_tables) == 1
                    else pa.concat_tables(splitter_prob_tables, promote_options="default")
                )
                if merged.num_rows != sp_table.num_rows:
                    raise RuntimeError(
                        "Aligned chunk row mismatch between main and group_splitter tables: "
                        f"{merged.num_rows} vs {sp_table.num_rows}"
                    )
                merged = self._merge_columns_from_table(merged, sp_table, list(sp_table.column_names))

            if endpoint_tables:
                ep_table = (
                    endpoint_tables[0]
                    if len(endpoint_tables) == 1
                    else pa.concat_tables(endpoint_tables, promote_options="default")
                )
                if merged.num_rows != ep_table.num_rows:
                    raise RuntimeError(
                        "Aligned chunk row mismatch between main and endpoint tables: "
                        f"{merged.num_rows} vs {ep_table.num_rows}"
                    )
                merged = self._merge_columns_from_table(merged, ep_table, list(ep_table.column_names))

            yield merged.combine_chunks()

    @staticmethod
    def _rename_columns(table: pa.Table, mapping: dict[str, str]) -> pa.Table:
        names = [mapping.get(name, name) for name in table.column_names]
        if names == table.column_names:
            return table
        return table.rename_columns(names)

    @staticmethod
    def _merge_columns_from_table(dst: pa.Table, src: pa.Table, columns: list[str]) -> pa.Table:
        out = dst
        for col in columns:
            if col in out.column_names:
                out = out.set_column(out.schema.get_field_index(col), col, src.column(col))
            else:
                out = out.append_column(col, src.column(col))
        return out

    @staticmethod
    def _shard_tasks(
        tasks: list[tuple[str, str | None, str | None, str | None, int]],
    ) -> list[tuple[str, str | None, str | None, str | None, int]]:
        worker = get_worker_info()
        if worker is None:
            return tasks
        return tasks[worker.id :: worker.num_workers]

    def _aligned_row_group_tasks(self) -> list[tuple[str, str | None, str | None, str | None, int]]:
        tasks: list[tuple[str, str | None, str | None, str | None, int]] = []
        for i, main_path in enumerate(self.parquet_paths):
            group_path = None if self.group_probs_parquet_paths is None else self.group_probs_parquet_paths[i]
            splitter_path = (
                None if self.group_splitter_parquet_paths is None else self.group_splitter_parquet_paths[i]
            )
            endpoint_path = None if self.endpoint_parquet_paths is None else self.endpoint_parquet_paths[i]

            main_pf = pq.ParquetFile(main_path)
            gp_pf = pq.ParquetFile(group_path) if group_path is not None else None
            sp_pf = pq.ParquetFile(splitter_path) if splitter_path is not None else None
            ep_pf = pq.ParquetFile(endpoint_path) if endpoint_path is not None else None

            if gp_pf is not None and gp_pf.num_row_groups != main_pf.num_row_groups:
                raise RuntimeError(
                    "Row-group mismatch between main and group_probs files: "
                    f"{main_path} ({main_pf.num_row_groups}) vs {group_path} ({gp_pf.num_row_groups})"
                )
            if sp_pf is not None and sp_pf.num_row_groups != main_pf.num_row_groups:
                raise RuntimeError(
                    "Row-group mismatch between main and group_splitter files: "
                    f"{main_path} ({main_pf.num_row_groups}) vs {splitter_path} ({sp_pf.num_row_groups})"
                )
            if ep_pf is not None and ep_pf.num_row_groups != main_pf.num_row_groups:
                raise RuntimeError(
                    "Row-group mismatch between main and endpoint files: "
                    f"{main_path} ({main_pf.num_row_groups}) vs {endpoint_path} ({ep_pf.num_row_groups})"
                )

            for rg in range(main_pf.num_row_groups):
                n_main = int(main_pf.metadata.row_group(rg).num_rows)
                if gp_pf is not None:
                    n_gp = int(gp_pf.metadata.row_group(rg).num_rows)
                    if n_gp != n_main:
                        raise RuntimeError(
                            f"Row count mismatch at row-group {rg}: {main_path} ({n_main}) vs {group_path} ({n_gp})"
                        )
                if sp_pf is not None:
                    n_sp = int(sp_pf.metadata.row_group(rg).num_rows)
                    if n_sp != n_main:
                        raise RuntimeError(
                            f"Row count mismatch at row-group {rg}: {main_path} ({n_main}) vs {splitter_path} ({n_sp})"
                        )
                if ep_pf is not None:
                    n_ep = int(ep_pf.metadata.row_group(rg).num_rows)
                    if n_ep != n_main:
                        raise RuntimeError(
                            f"Row count mismatch at row-group {rg}: {main_path} ({n_main}) vs {endpoint_path} ({n_ep})"
                        )
                tasks.append((main_path, group_path, splitter_path, endpoint_path, rg))
        return tasks

    def _extract_chunk_inputs(self, table: pa.Table) -> dict[str, np.ndarray]:
        hits_tg_off, hits_tg_vals = self._extract_list_column(table, "hits_time_group", np.int64)
        _, hits_coord_vals = self._extract_list_column(table, "hits_coord", np.float32)
        _, hits_z_vals = self._extract_list_column(table, "hits_z", np.float32)
        _, hits_edep_vals = self._extract_list_column(table, "hits_edep", np.float32)
        _, hits_view_vals = self._extract_list_column(table, "hits_strip_type", np.int32)

        out: dict[str, np.ndarray] = {
            "event_ids": self._to_np(table.column("event_id").chunk(0), np.int64),
            "hits_time_group_offsets": hits_tg_off,
            "hits_time_group_values": hits_tg_vals,
            "hits_coord_values": hits_coord_vals,
            "hits_z_values": hits_z_vals,
            "hits_edep_values": hits_edep_vals,
            "hits_strip_type_values": hits_view_vals,
        }

        has_particle_mask = "hits_particle_mask" in table.column_names
        out["has_hits_particle_mask"] = np.asarray([1 if has_particle_mask else 0], dtype=np.int8)
        if has_particle_mask:
            pm_off, pm_vals = self._extract_list_column(table, "hits_particle_mask", np.int32)
            out["hits_particle_mask_offsets"] = pm_off
            out["hits_particle_mask_values"] = pm_vals

        has_mc_contrib = "hits_contrib_mc_event_id" in table.column_names
        out["has_hits_contrib_mc_event_id"] = np.asarray([1 if has_mc_contrib else 0], dtype=np.int8)
        if has_mc_contrib:
            contrib_outer_off, contrib_off, contrib_vals = self._extract_nested_list_column(
                table, "hits_contrib_mc_event_id", np.int32
            )
            out["hits_contrib_mc_event_id_outer_offsets"] = contrib_outer_off
            out["hits_contrib_mc_event_id_offsets"] = contrib_off
            out["hits_contrib_mc_event_id_values"] = contrib_vals

        has_group_probs = all(col in table.column_names for col in self.GROUP_PROB_COLUMNS)
        out["has_group_prob_columns"] = np.asarray([1 if has_group_probs else 0], dtype=np.int8)
        if has_group_probs:
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

        has_splitter_probs = all(col in table.column_names for col in self.SPLITTER_PROB_COLUMNS[:3])
        splitter_tg_col = None
        if "splitter_time_group_ids" in table.column_names:
            splitter_tg_col = "splitter_time_group_ids"
        elif "time_group_ids" in table.column_names and "endpoint_time_group_ids" not in table.column_names:
            splitter_tg_col = "time_group_ids"
        out["has_splitter_prob_columns"] = np.asarray([1 if has_splitter_probs else 0], dtype=np.int8)
        out["has_splitter_time_group_ids"] = np.asarray([1 if splitter_tg_col is not None else 0], dtype=np.int8)
        if has_splitter_probs:
            hp_off, hp_vals = self._extract_list_column(table, "pred_hit_pion", np.float32)
            hm_off, hm_vals = self._extract_list_column(table, "pred_hit_muon", np.float32)
            hi_off, hi_vals = self._extract_list_column(table, "pred_hit_mip", np.float32)
            out.update(
                {
                    "pred_hit_pion_offsets": hp_off,
                    "pred_hit_pion_values": hp_vals,
                    "pred_hit_muon_offsets": hm_off,
                    "pred_hit_muon_values": hm_vals,
                    "pred_hit_mip_offsets": hi_off,
                    "pred_hit_mip_values": hi_vals,
                }
            )
        if splitter_tg_col is not None:
            stg_off, stg_vals = self._extract_list_column(table, splitter_tg_col, np.int64)
            out["splitter_time_group_offsets"] = stg_off
            out["splitter_time_group_values"] = stg_vals

        endpoint_quant_cols = self._endpoint_quantile_columns()
        has_endpoint_quant = all(col in table.column_names for col in endpoint_quant_cols)
        has_endpoint_base = all(col in table.column_names for col in self.ENDPOINT_BASE_COLUMNS)
        endpoint_tg_col = None
        if "endpoint_time_group_ids" in table.column_names:
            endpoint_tg_col = "endpoint_time_group_ids"
        elif (
            "time_group_ids" in table.column_names
            and "splitter_time_group_ids" not in table.column_names
            and not has_splitter_probs
        ):
            endpoint_tg_col = "time_group_ids"

        out["has_endpoint_quantile_columns"] = np.asarray([1 if has_endpoint_quant else 0], dtype=np.int8)
        out["has_endpoint_base_columns"] = np.asarray([1 if has_endpoint_base else 0], dtype=np.int8)
        out["has_endpoint_time_group_ids"] = np.asarray([1 if endpoint_tg_col is not None else 0], dtype=np.int8)

        if has_endpoint_quant:
            first = endpoint_quant_cols[0]
            e_off, e_vals = self._extract_list_column(table, first, np.float32)
            out[f"{first}_offsets"] = e_off
            out[f"{first}_values"] = e_vals
            out["endpoint_count_offsets"] = e_off
            for col in endpoint_quant_cols[1:]:
                col_off, col_vals = self._extract_list_column(table, col, np.float32)
                out[f"{col}_offsets"] = col_off
                out[f"{col}_values"] = col_vals
        elif has_endpoint_base:
            first = self.ENDPOINT_BASE_COLUMNS[0]
            e_off, e_vals = self._extract_list_column(table, first, np.float32)
            out[f"{first}_offsets"] = e_off
            out[f"{first}_values"] = e_vals
            out["endpoint_count_offsets"] = e_off
            for col in self.ENDPOINT_BASE_COLUMNS[1:]:
                col_off, col_vals = self._extract_list_column(table, col, np.float32)
                out[f"{col}_offsets"] = col_off
                out[f"{col}_values"] = col_vals

        if endpoint_tg_col is not None:
            etg_off, etg_vals = self._extract_list_column(table, endpoint_tg_col, np.int64)
            out["endpoint_time_group_offsets"] = etg_off
            out["endpoint_time_group_values"] = etg_vals

        return out

    @staticmethod
    def _counts_from_offsets(offsets: np.ndarray) -> np.ndarray:
        return (offsets[1:] - offsets[:-1]).astype(np.int64, copy=False)

    @classmethod
    def _compute_group_counts_from_hits(
        cls,
        *,
        n_rows: int,
        hit_offsets: np.ndarray,
        hit_time_group_values: np.ndarray,
        hit_counts: np.ndarray,
    ) -> np.ndarray:
        if hit_time_group_values.size == 0:
            return np.zeros((n_rows,), dtype=np.int64)
        starts = hit_offsets[:-1].astype(np.int64, copy=False)
        safe_starts = np.minimum(starts, max(0, hit_time_group_values.size - 1))
        row_tg_max = np.maximum.reduceat(hit_time_group_values, safe_starts).astype(np.int64, copy=False)
        row_tg_max[hit_counts == 0] = -1
        return row_tg_max + 1

    @classmethod
    def _compute_event_layout(
        cls,
        *,
        n_rows: int,
        hits_time_group_offsets: np.ndarray,
        hits_time_group_values: np.ndarray,
        row_group_count_candidates: list[np.ndarray],
    ) -> dict[str, np.ndarray | int]:
        hit_counts = cls._counts_from_offsets(hits_time_group_offsets)
        total_nodes = int(hit_counts.sum())
        node_ptr = np.zeros((n_rows + 1,), dtype=np.int64)
        node_ptr[1:] = np.cumsum(hit_counts)

        edge_counts = hit_counts * np.maximum(hit_counts - 1, 0)
        total_edges = int(edge_counts.sum())
        edge_ptr = np.zeros((n_rows + 1,), dtype=np.int64)
        edge_ptr[1:] = np.cumsum(edge_counts)

        row_ids_hit = np.repeat(np.arange(n_rows, dtype=np.int64), hit_counts)
        hits_group_counts = cls._compute_group_counts_from_hits(
            n_rows=n_rows,
            hit_offsets=hits_time_group_offsets,
            hit_time_group_values=hits_time_group_values,
            hit_counts=hit_counts,
        )

        row_group_counts = np.maximum.reduce([hits_group_counts, *row_group_count_candidates]).astype(
            np.int64, copy=False
        )
        group_ptr = np.zeros((n_rows + 1,), dtype=np.int64)
        group_ptr[1:] = np.cumsum(row_group_counts)
        total_groups = int(group_ptr[-1])
        row_group_base = group_ptr[:-1]

        if total_nodes > 0 and total_groups > 0:
            global_group_id = row_group_base[row_ids_hit] + hits_time_group_values
        else:
            global_group_id = np.zeros((0,), dtype=np.int64)

        return {
            "hit_counts": hit_counts,
            "total_nodes": total_nodes,
            "row_ids_hit": row_ids_hit,
            "hits_group_counts": hits_group_counts,
            "row_group_counts": row_group_counts,
            "group_ptr": group_ptr,
            "total_groups": total_groups,
            "row_group_base": row_group_base,
            "global_group_id": global_group_id,
            "edge_counts": edge_counts,
            "total_edges": total_edges,
            "node_ptr": node_ptr,
            "edge_ptr": edge_ptr,
        }

    @staticmethod
    def _particle_mask_to_multihot(mask_values: np.ndarray) -> np.ndarray:
        out = np.zeros((mask_values.size, 3), dtype=np.float32)
        out[:, 0] = ((mask_values & 1) != 0).astype(np.float32, copy=False)
        out[:, 1] = ((mask_values & 2) != 0).astype(np.float32, copy=False)
        out[:, 2] = ((mask_values & 4) != 0).astype(np.float32, copy=False)
        return out

    @classmethod
    def _extract_nested_list_column(cls, table: pa.Table, name: str, dtype) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        arr = table.column(name).chunk(0)
        outer_offsets = cls._to_np(arr.offsets, np.int64)
        inner = arr.values
        inner_offsets = cls._to_np(inner.offsets, np.int64)
        inner_values = cls._to_np(inner.values, dtype)
        return outer_offsets, inner_offsets, inner_values

    @staticmethod
    def _fill_group_column_from_group_values(
        *,
        out: np.ndarray,
        dst_col: int,
        vals: np.ndarray,
        offs: np.ndarray,
        total_groups: int,
        local_gid: np.ndarray,
        row_ids_group: np.ndarray,
    ) -> None:
        if total_groups == 0 or vals.size == 0:
            return
        counts = (offs[1:] - offs[:-1]).astype(np.int64, copy=False)
        valid = local_gid < counts[row_ids_group]
        if not np.any(valid):
            return
        idx = offs[row_ids_group[valid]] + local_gid[valid]
        out[valid, dst_col] = vals[idx].astype(np.float32, copy=False)

    @classmethod
    def _fill_group_probs_from_lists(
        cls,
        *,
        group_probs_out: np.ndarray,
        total_groups: int,
        local_gid: np.ndarray,
        row_ids_group: np.ndarray,
        pred_pion_values: np.ndarray,
        pred_pion_offsets: np.ndarray,
        pred_muon_values: np.ndarray,
        pred_muon_offsets: np.ndarray,
        pred_mip_values: np.ndarray,
        pred_mip_offsets: np.ndarray,
    ) -> None:
        cls._fill_group_column_from_group_values(
            out=group_probs_out,
            dst_col=0,
            vals=pred_pion_values,
            offs=pred_pion_offsets,
            total_groups=total_groups,
            local_gid=local_gid,
            row_ids_group=row_ids_group,
        )
        cls._fill_group_column_from_group_values(
            out=group_probs_out,
            dst_col=1,
            vals=pred_muon_values,
            offs=pred_muon_offsets,
            total_groups=total_groups,
            local_gid=local_gid,
            row_ids_group=row_ids_group,
        )
        cls._fill_group_column_from_group_values(
            out=group_probs_out,
            dst_col=2,
            vals=pred_mip_values,
            offs=pred_mip_offsets,
            total_groups=total_groups,
            local_gid=local_gid,
            row_ids_group=row_ids_group,
        )

    @staticmethod
    def _build_splitter_value_index_per_hit(
        *,
        n_rows: int,
        hit_offsets: np.ndarray,
        hit_time_group_values: np.ndarray,
        splitter_offsets: np.ndarray,
        splitter_time_group_offsets: np.ndarray | None,
        splitter_time_group_values: np.ndarray | None,
    ) -> np.ndarray:
        total_hits = int(hit_offsets[-1]) if hit_offsets.size > 0 else 0
        out = np.zeros((total_hits,), dtype=np.int64)
        for row in range(n_rows):
            h0 = int(hit_offsets[row])
            h1 = int(hit_offsets[row + 1])
            n_hits = h1 - h0
            s0 = int(splitter_offsets[row])
            s1 = int(splitter_offsets[row + 1])
            if (s1 - s0) != n_hits:
                raise RuntimeError("Splitter probability list length does not match hits.")
            if n_hits <= 0:
                continue

            local_map = s0 + np.arange(n_hits, dtype=np.int64)
            if splitter_time_group_offsets is not None and splitter_time_group_values is not None:
                tg0 = int(splitter_time_group_offsets[row])
                tg1 = int(splitter_time_group_offsets[row + 1])
                if (tg1 - tg0) != n_hits:
                    raise RuntimeError("Splitter time_group_ids length does not match hits.")
                hit_tg = hit_time_group_values[h0:h1]
                split_tg = splitter_time_group_values[tg0:tg1]
                if not np.array_equal(hit_tg, split_tg):
                    for gid in np.unique(hit_tg):
                        hit_pos = np.flatnonzero(hit_tg == gid)
                        split_pos = np.flatnonzero(split_tg == gid)
                        if hit_pos.size != split_pos.size:
                            raise RuntimeError(
                                "Splitter time_group_ids are not aligned with event hits."
                            )
                        local_map[hit_pos] = s0 + split_pos
            out[h0:h1] = local_map
        return out

    @staticmethod
    def _build_group_value_index(
        *,
        n_rows: int,
        row_group_counts: np.ndarray,
        row_group_base: np.ndarray,
        value_offsets: np.ndarray,
        group_ids_offsets: np.ndarray | None,
        group_ids_values: np.ndarray | None,
    ) -> np.ndarray:
        total_groups = int(np.sum(row_group_counts, dtype=np.int64))
        out = np.full((total_groups,), -1, dtype=np.int64)
        for row in range(n_rows):
            gcount = int(row_group_counts[row])
            if gcount <= 0:
                continue
            v0 = int(value_offsets[row])
            v1 = int(value_offsets[row + 1])
            n_vals = v1 - v0
            if n_vals <= 0:
                continue
            base = int(row_group_base[row])
            if group_ids_offsets is not None and group_ids_values is not None:
                t0 = int(group_ids_offsets[row])
                t1 = int(group_ids_offsets[row + 1])
                if (t1 - t0) != n_vals:
                    raise RuntimeError("Endpoint time_group_ids length does not match endpoint predictions.")
                gids = group_ids_values[t0:t1]
                valid = (gids >= 0) & (gids < gcount)
                if np.any(valid):
                    local_idx = np.flatnonzero(valid).astype(np.int64, copy=False)
                    out[base + gids[valid]] = v0 + local_idx
            else:
                upto = min(gcount, n_vals)
                if upto <= 0:
                    continue
                rel = np.arange(upto, dtype=np.int64)
                out[base + rel] = v0 + rel
        return out

    @staticmethod
    def _populate_node_tensors(
        *,
        hit_coord_values: np.ndarray,
        hit_z_values: np.ndarray,
        hit_edep_values: np.ndarray,
        hit_strip_type_values: np.ndarray,
        hit_time_group_values: np.ndarray,
        x_out: np.ndarray,
        time_group_ids_out: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_out[:, 0] = hit_coord_values
        x_out[:, 1] = hit_z_values
        x_out[:, 2] = hit_edep_values
        x_out[:, 3] = hit_strip_type_values.astype(np.float32, copy=False)
        time_group_ids_out[:] = hit_time_group_values
        return x_out[:, 0], x_out[:, 1], x_out[:, 2], hit_strip_type_values

    def _populate_event_edges(
        self,
        *,
        hit_counts: np.ndarray,
        node_ptr: np.ndarray,
        edge_ptr: np.ndarray,
        sorted_coord_values: np.ndarray,
        sorted_z_values: np.ndarray,
        sorted_edep_values: np.ndarray,
        sorted_strip_type_values: np.ndarray,
        global_group_id: np.ndarray,
        mc_event_id_offsets: np.ndarray | None,
        mc_event_id_values: np.ndarray | None,
        edge_index_out: np.ndarray,
        edge_attr_out: np.ndarray,
        targets_out: np.ndarray | None,
    ) -> None:
        edge_tpl_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for k in np.unique(hit_counts):
            k = int(k)
            if k <= 1:
                continue
            src, dst = self._complete_digraph_cached(k, edge_tpl_cache)
            ecount = int(src.shape[0])
            events = np.flatnonzero(hit_counts == k)
            if events.size == 0:
                continue

            rel_edge = np.arange(ecount, dtype=np.int64)
            graph_block = int(self.edge_populate_graph_block)
            for i in range(0, events.size, graph_block):
                g = events[i : i + graph_block]
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
                edge_attr_out[pos, 4] = (global_group_id[src_idx] == global_group_id[dst_idx]).astype(np.float32)

                if targets_out is not None:
                    if mc_event_id_offsets is None or mc_event_id_values is None:
                        raise RuntimeError("Event-splitter target construction requires hits_contrib_mc_event_id.")
                    local_nodes = np.arange(k, dtype=np.int64)
                    for block_idx, base in enumerate(node_base):
                        edge_base_i = int(edge_base[block_idx])
                        labels = self._build_edge_labels_from_mc_event_ids(
                            node_indices=(int(base) + local_nodes),
                            src_local=src,
                            dst_local=dst,
                            mc_event_id_offsets=mc_event_id_offsets,
                            mc_event_id_values=mc_event_id_values,
                        )
                        targets_out[edge_base_i : edge_base_i + ecount, 0] = labels

    @staticmethod
    def _build_edge_labels_from_mc_event_ids(
        *,
        node_indices: np.ndarray,
        src_local: np.ndarray,
        dst_local: np.ndarray,
        mc_event_id_offsets: np.ndarray,
        mc_event_id_values: np.ndarray,
    ) -> np.ndarray:
        """Build directed edge labels for one event by MC contributor overlap."""
        k = int(node_indices.shape[0])
        if k <= 1:
            return np.zeros((src_local.shape[0],), dtype=np.float32)

        mc_to_nodes: dict[int, list[int]] = {}
        for local_i, node_idx in enumerate(node_indices):
            s0 = int(mc_event_id_offsets[node_idx])
            s1 = int(mc_event_id_offsets[node_idx + 1])
            if s1 <= s0:
                continue
            ids = np.unique(mc_event_id_values[s0:s1])
            for mc_id in ids:
                key = int(mc_id)
                if key not in mc_to_nodes:
                    mc_to_nodes[key] = [local_i]
                else:
                    mc_to_nodes[key].append(local_i)

        if not mc_to_nodes:
            return np.zeros((src_local.shape[0],), dtype=np.float32)

        adjacency = np.zeros((k, k), dtype=np.bool_)
        for nodes in mc_to_nodes.values():
            if len(nodes) <= 1:
                continue
            idx = np.asarray(nodes, dtype=np.int64)
            adjacency[np.ix_(idx, idx)] = True

        np.fill_diagonal(adjacency, False)
        return adjacency[src_local, dst_local].astype(np.float32, copy=False)

    def _build_chunk_graph_arrays(self, *, table: pa.Table) -> dict:
        n_rows = int(table.num_rows)
        chunk_in = self._extract_chunk_inputs(table)

        row_group_count_candidates: list[np.ndarray] = []
        if self.use_group_probs and int(chunk_in["has_group_prob_columns"][0]) == 1:
            row_group_count_candidates.append(self._counts_from_offsets(chunk_in["pred_pion_offsets"]))
        if self.use_endpoint_preds and "endpoint_count_offsets" in chunk_in:
            row_group_count_candidates.append(self._counts_from_offsets(chunk_in["endpoint_count_offsets"]))

        layout = self._compute_event_layout(
            n_rows=n_rows,
            hits_time_group_offsets=chunk_in["hits_time_group_offsets"],
            hits_time_group_values=chunk_in["hits_time_group_values"],
            row_group_count_candidates=row_group_count_candidates,
        )

        hit_counts = layout["hit_counts"]
        total_nodes = int(layout["total_nodes"])
        total_edges = int(layout["total_edges"])
        total_groups = int(layout["total_groups"])
        node_ptr = layout["node_ptr"]
        edge_ptr = layout["edge_ptr"]
        group_ptr = layout["group_ptr"]
        row_group_counts = layout["row_group_counts"]
        row_group_base = layout["row_group_base"]
        global_group_id = layout["global_group_id"]

        x_out = np.empty((total_nodes, self.NODE_FEATURE_DIM), dtype=np.float32)
        edge_index_out = np.empty((2, total_edges), dtype=np.int64)
        edge_attr_out = np.empty((total_edges, self.EDGE_FEATURE_DIM), dtype=np.float32)
        tgroup_out = np.empty((total_nodes,), dtype=np.int64)

        if total_groups > 0:
            group_ids = np.arange(total_groups, dtype=np.int64)
            row_ids_group = np.repeat(np.arange(n_rows, dtype=np.int64), row_group_counts)
            local_gid = group_ids - row_group_base[row_ids_group]
        else:
            row_ids_group = np.zeros((0,), dtype=np.int64)
            local_gid = np.zeros((0,), dtype=np.int64)

        coord_vals, z_vals, e_vals, view_vals = self._populate_node_tensors(
            hit_coord_values=chunk_in["hits_coord_values"],
            hit_z_values=chunk_in["hits_z_values"],
            hit_edep_values=chunk_in["hits_edep_values"],
            hit_strip_type_values=chunk_in["hits_strip_type_values"],
            hit_time_group_values=chunk_in["hits_time_group_values"],
            x_out=x_out,
            time_group_ids_out=tgroup_out,
        )

        has_particle_mask = int(chunk_in["has_hits_particle_mask"][0]) == 1
        has_mc_contrib = int(chunk_in.get("has_hits_contrib_mc_event_id", np.asarray([0], dtype=np.int8))[0]) == 1
        particle_mask_values = np.zeros((total_nodes,), dtype=np.int32)
        node_truth = np.zeros((total_nodes, self.NUM_CLASSES), dtype=np.float32)
        group_truth = np.zeros((total_groups, self.NUM_CLASSES), dtype=np.float32)

        if has_particle_mask:
            pm_counts = self._counts_from_offsets(chunk_in["hits_particle_mask_offsets"])
            if not np.array_equal(pm_counts, hit_counts):
                raise RuntimeError("hits_particle_mask list lengths do not match hit arrays.")
            particle_mask_values = chunk_in["hits_particle_mask_values"].astype(np.int32, copy=False)
            node_truth[:, :] = self._particle_mask_to_multihot(particle_mask_values)
            if total_groups > 0:
                for cls_id in range(self.NUM_CLASSES):
                    cls_mask = node_truth[:, cls_id] > 0.5
                    if not np.any(cls_mask):
                        continue
                    present = np.bincount(
                        global_group_id[cls_mask],
                        minlength=total_groups,
                    ) > 0
                    group_truth[:, cls_id] = present.astype(np.float32, copy=False)

        group_probs_out = np.zeros((total_groups, self.NUM_CLASSES), dtype=np.float32)
        if self.use_group_probs:
            if int(chunk_in["has_group_prob_columns"][0]) == 1:
                self._fill_group_probs_from_lists(
                    group_probs_out=group_probs_out,
                    total_groups=total_groups,
                    local_gid=local_gid,
                    row_ids_group=row_ids_group,
                    pred_pion_values=chunk_in["pred_pion_values"],
                    pred_pion_offsets=chunk_in["pred_pion_offsets"],
                    pred_muon_values=chunk_in["pred_muon_values"],
                    pred_muon_offsets=chunk_in["pred_muon_offsets"],
                    pred_mip_values=chunk_in["pred_mip_values"],
                    pred_mip_offsets=chunk_in["pred_mip_offsets"],
                )
            else:
                group_probs_out[:] = group_truth

        splitter_probs_out = np.zeros((total_nodes, self.NUM_CLASSES), dtype=np.float32)
        if self.use_splitter_probs:
            if int(chunk_in["has_splitter_prob_columns"][0]) == 1:
                splitter_counts = self._counts_from_offsets(chunk_in["pred_hit_pion_offsets"])
                if not np.array_equal(splitter_counts, hit_counts):
                    raise RuntimeError("Splitter probability list lengths do not match hit counts.")
                has_splitter_tg = int(chunk_in["has_splitter_time_group_ids"][0]) == 1
                splitter_index_for_hit = self._build_splitter_value_index_per_hit(
                    n_rows=n_rows,
                    hit_offsets=chunk_in["hits_time_group_offsets"],
                    hit_time_group_values=chunk_in["hits_time_group_values"],
                    splitter_offsets=chunk_in["pred_hit_pion_offsets"],
                    splitter_time_group_offsets=(
                        chunk_in["splitter_time_group_offsets"] if has_splitter_tg else None
                    ),
                    splitter_time_group_values=(
                        chunk_in["splitter_time_group_values"] if has_splitter_tg else None
                    ),
                )
                idx = splitter_index_for_hit
                splitter_probs_out[:, 0] = chunk_in["pred_hit_pion_values"][idx].astype(np.float32, copy=False)
                splitter_probs_out[:, 1] = chunk_in["pred_hit_muon_values"][idx].astype(np.float32, copy=False)
                splitter_probs_out[:, 2] = chunk_in["pred_hit_mip_values"][idx].astype(np.float32, copy=False)
            else:
                splitter_probs_out[:] = node_truth

        endpoint_preds_out = np.zeros((total_groups, self.ENDPOINT_DIM), dtype=np.float32)
        if self.use_endpoint_preds and "endpoint_count_offsets" in chunk_in and total_groups > 0:
            has_endpoint_tg = int(chunk_in["has_endpoint_time_group_ids"][0]) == 1
            value_index = self._build_group_value_index(
                n_rows=n_rows,
                row_group_counts=row_group_counts,
                row_group_base=row_group_base,
                value_offsets=chunk_in["endpoint_count_offsets"],
                group_ids_offsets=(chunk_in["endpoint_time_group_offsets"] if has_endpoint_tg else None),
                group_ids_values=(chunk_in["endpoint_time_group_values"] if has_endpoint_tg else None),
            )
            valid = value_index >= 0
            if np.any(valid):
                for c_idx, base in enumerate(self.ENDPOINT_BASE_COLUMNS):
                    if int(chunk_in["has_endpoint_quantile_columns"][0]) == 1:
                        for q_idx, suffix in enumerate(self.ENDPOINT_QUANTILE_SUFFIXES):
                            col_name = f"{base}_{suffix}"
                            vals = chunk_in[f"{col_name}_values"]
                            endpoint_preds_out[valid, (c_idx * 3) + q_idx] = vals[value_index[valid]].astype(
                                np.float32, copy=False
                            )
                    elif int(chunk_in["has_endpoint_base_columns"][0]) == 1:
                        vals = chunk_in[f"{base}_values"]
                        replicated = vals[value_index[valid]].astype(np.float32, copy=False)
                        endpoint_preds_out[valid, (c_idx * 3) + 0] = replicated
                        endpoint_preds_out[valid, (c_idx * 3) + 1] = replicated
                        endpoint_preds_out[valid, (c_idx * 3) + 2] = replicated

        mc_event_id_offsets: np.ndarray | None = None
        mc_event_id_values: np.ndarray | None = None
        if has_mc_contrib:
            contrib_outer_counts = self._counts_from_offsets(chunk_in["hits_contrib_mc_event_id_outer_offsets"])
            if not np.array_equal(contrib_outer_counts, hit_counts):
                raise RuntimeError("hits_contrib_mc_event_id outer lengths do not match hit arrays.")
            mc_event_id_offsets = chunk_in["hits_contrib_mc_event_id_offsets"]
            mc_event_id_values = chunk_in["hits_contrib_mc_event_id_values"]

        targets_out: np.ndarray | None = None
        if self.include_targets:
            if not has_mc_contrib:
                raise RuntimeError("Training mode requires hits_contrib_mc_event_id for event-splitter targets.")
            targets_out = np.zeros((total_edges, 1), dtype=np.float32)

        self._populate_event_edges(
            hit_counts=hit_counts,
            node_ptr=node_ptr,
            edge_ptr=edge_ptr,
            sorted_coord_values=coord_vals,
            sorted_z_values=z_vals,
            sorted_edep_values=e_vals,
            sorted_strip_type_values=view_vals,
            global_group_id=global_group_id,
            mc_event_id_offsets=mc_event_id_offsets,
            mc_event_id_values=mc_event_id_values,
            edge_index_out=edge_index_out,
            edge_attr_out=edge_attr_out,
            targets_out=targets_out,
        )

        chunk_out = {
            "x": torch.from_numpy(x_out),
            "edge_index": torch.from_numpy(edge_index_out),
            "edge_attr": torch.from_numpy(edge_attr_out),
            "group_ptr": torch.from_numpy(group_ptr),
            "time_group_ids": torch.from_numpy(tgroup_out),
            "group_probs": torch.from_numpy(group_probs_out),
            "splitter_probs": torch.from_numpy(splitter_probs_out),
            "endpoint_preds": torch.from_numpy(endpoint_preds_out),
            "graph_event_ids": torch.from_numpy(np.arange(n_rows, dtype=np.int64)),
            "node_ptr": torch.from_numpy(node_ptr),
            "edge_ptr": torch.from_numpy(edge_ptr),
            "num_rows": n_rows,
            "num_graphs": int(n_rows),
            "num_groups": int(total_groups),
        }
        if targets_out is not None:
            chunk_out["targets"] = torch.from_numpy(targets_out)
        return chunk_out

    @staticmethod
    def _slice_chunk_batch(chunk: dict, g0: int, g1: int) -> Data:
        node_ptr = chunk["node_ptr"]
        edge_ptr = chunk["edge_ptr"]
        group_ptr = chunk["group_ptr"]

        n0 = int(node_ptr[g0].item())
        n1 = int(node_ptr[g1].item())
        e0 = int(edge_ptr[g0].item())
        e1 = int(edge_ptr[g1].item())
        gp0 = int(group_ptr[g0].item())
        gp1 = int(group_ptr[g1].item())

        d = Data(
            x=chunk["x"][n0:n1],
            edge_index=(chunk["edge_index"][:, e0:e1] - n0),
            edge_attr=chunk["edge_attr"][e0:e1],
            group_ptr=(chunk["group_ptr"][g0 : g1 + 1] - gp0),
            time_group_ids=chunk["time_group_ids"][n0:n1],
            group_probs=chunk["group_probs"][gp0:gp1],
            splitter_probs=chunk["splitter_probs"][n0:n1],
            endpoint_preds=chunk["endpoint_preds"][gp0:gp1],
        )
        if "targets" in chunk:
            d.y = chunk["targets"][e0:e1]

        local_counts = (node_ptr[g0 + 1 : g1 + 1] - node_ptr[g0:g1]).to(dtype=torch.int64)
        d.batch = torch.repeat_interleave(torch.arange(g1 - g0, dtype=torch.int64), local_counts)

        d.node_ptr = chunk["node_ptr"][g0 : g1 + 1] - n0
        d.edge_ptr = chunk["edge_ptr"][g0 : g1 + 1] - e0
        d.event_ids = chunk["graph_event_ids"][g0:g1]
        d.num_graphs = int(g1 - g0)
        d.num_groups = int(gp1 - gp0)
        return d
