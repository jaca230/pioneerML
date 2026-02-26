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


class PositronAngleGraphLoader(TimeGroupGraphLoader):
    """Chunked graph loader for positron momentum-vector quantile regression."""

    GRAPH_COLUMNS = [
        "event_id",
        "hits_time_group",
        "hits_strip_type",
        "hits_coord",
        "hits_z",
        "hits_edep",
    ]
    OPTIONAL_TRUTH_COLUMNS = ["hits_particle_mask", "hits_pdg_id"]
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
    EVENT_SPLITTER_COLUMNS = [
        "time_group_ids",
        "edge_src_index",
        "edge_dst_index",
        "pred_edge_affinity",
    ]
    PION_STOP_COLUMNS = ["pred_pion_stop_x_q50", "pred_pion_stop_y_q50", "pred_pion_stop_z_q50", "time_group_ids"]

    NODE_FEATURE_DIM = 4
    EDGE_FEATURE_DIM = 4
    NUM_CLASSES = 3
    ENDPOINT_DIM = 18
    EVENT_AFFINITY_DIM = 3
    PION_STOP_DIM = 3
    NUM_TARGET_QUANTILES = 3

    MODE_TRAIN = "train"
    MODE_INFERENCE = "inference"

    TARGET_COLUMNS = ("positron_px", "positron_py", "positron_pz")
    TARGET_BASE_DIM = 3
    TARGET_DIM = 9
    MIN_RELEVANT_HITS = 2
    POSITRON_PDG_ID = -11

    def __init__(
        self,
        parquet_paths: list[str],
        *,
        group_probs_parquet_paths: list[str] | None = None,
        group_splitter_parquet_paths: list[str] | None = None,
        endpoint_parquet_paths: list[str] | None = None,
        event_splitter_parquet_paths: list[str] | None = None,
        pion_stop_parquet_paths: list[str] | None = None,
        mode: str = MODE_TRAIN,
        use_group_probs: bool = True,
        use_splitter_probs: bool = True,
        use_endpoint_preds: bool = True,
        use_event_splitter_affinity: bool = True,
        use_pion_stop_preds: bool = False,
        training_relevant_only: bool = True,
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
        if not self.TARGET_COLUMNS:
            raise ValueError(f"{self.__class__.__name__} must define TARGET_COLUMNS.")
        if int(self.TARGET_BASE_DIM) <= 0:
            raise ValueError(f"{self.__class__.__name__} must define TARGET_BASE_DIM > 0.")
        expected_target_dim = int(self.TARGET_BASE_DIM * self.NUM_TARGET_QUANTILES)
        if int(self.TARGET_DIM) != expected_target_dim:
            raise ValueError(
                f"{self.__class__.__name__}.TARGET_DIM ({self.TARGET_DIM}) must equal "
                f"TARGET_BASE_DIM*NUM_TARGET_QUANTILES ({expected_target_dim})."
            )

        self._parquet_paths_init = [str(Path(p).expanduser().resolve()) for p in parquet_paths]
        mode_norm = str(mode).strip().lower()
        if mode_norm not in (self.MODE_TRAIN, self.MODE_INFERENCE):
            raise ValueError(f"Unsupported mode: {mode}. Expected 'train' or 'inference'.")
        self.mode = mode_norm

        self.use_group_probs = bool(use_group_probs)
        self.use_splitter_probs = bool(use_splitter_probs)
        self.use_endpoint_preds = bool(use_endpoint_preds)
        self.use_event_splitter_affinity = bool(use_event_splitter_affinity)
        self.use_pion_stop_preds = bool(use_pion_stop_preds)
        self.training_relevant_only = bool(training_relevant_only)

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
        self.event_splitter_parquet_paths = self._resolve_optional_paths(
            parquet_paths=parquet_paths,
            secondary_paths=event_splitter_parquet_paths,
            field_name="event_splitter_parquet_paths",
        )
        self.pion_stop_parquet_paths = self._resolve_optional_paths(
            parquet_paths=parquet_paths,
            secondary_paths=pion_stop_parquet_paths,
            field_name="pion_stop_parquet_paths",
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

    def _required_columns_for_mode(self, mode: str) -> list[str]:
        if mode == self.MODE_TRAIN:
            return [*self.GRAPH_COLUMNS, *self.OPTIONAL_TRUTH_COLUMNS, *list(self.TARGET_COLUMNS)]
        return list(self.GRAPH_COLUMNS)

    def _resolve_default_columns(self, required: list[str]) -> list[str]:
        out = list(required)

        def _extend_unique(names: list[str]) -> None:
            for name in names:
                if name not in out:
                    out.append(name)

        if self.group_probs_parquet_paths is None and self.use_group_probs:
            if self._columns_present_in_all_files(self.GROUP_PROB_COLUMNS):
                _extend_unique(self.GROUP_PROB_COLUMNS)

        if self.group_splitter_parquet_paths is None and self.use_splitter_probs:
            if self._columns_present_in_all_files(self.SPLITTER_PROB_COLUMNS[:3]):
                _extend_unique(self.SPLITTER_PROB_COLUMNS[:3])
                if self._columns_present_in_all_files(["time_group_ids"]):
                    _extend_unique(["time_group_ids"])

        if self.endpoint_parquet_paths is None and self.use_endpoint_preds:
            quant_cols = self._endpoint_quantile_columns()
            if self._columns_present_in_all_files(quant_cols):
                _extend_unique(quant_cols)
            elif self._columns_present_in_all_files(self.ENDPOINT_BASE_COLUMNS):
                _extend_unique(self.ENDPOINT_BASE_COLUMNS)
            if self._columns_present_in_all_files(["time_group_ids"]):
                _extend_unique(["time_group_ids"])

        if self.event_splitter_parquet_paths is None and self.use_event_splitter_affinity:
            if self._columns_present_in_all_files(["edge_src_index", "edge_dst_index", "pred_edge_affinity"]):
                _extend_unique(["edge_src_index", "edge_dst_index", "pred_edge_affinity"])
                if self._columns_present_in_all_files(["time_group_ids"]):
                    _extend_unique(["time_group_ids"])

        if self.pion_stop_parquet_paths is None and self.use_pion_stop_preds:
            if self._columns_present_in_all_files(self.PION_STOP_COLUMNS[:3]):
                _extend_unique(self.PION_STOP_COLUMNS[:3])
                if self._columns_present_in_all_files(["time_group_ids"]):
                    _extend_unique(["time_group_ids"])

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
        kwargs["event_splitter_parquet_paths"] = self.event_splitter_parquet_paths
        kwargs["pion_stop_parquet_paths"] = self.pion_stop_parquet_paths
        kwargs["mode"] = self.mode
        kwargs["use_group_probs"] = self.use_group_probs
        kwargs["use_splitter_probs"] = self.use_splitter_probs
        kwargs["use_endpoint_preds"] = self.use_endpoint_preds
        kwargs["use_event_splitter_affinity"] = self.use_event_splitter_affinity
        kwargs["use_pion_stop_preds"] = self.use_pion_stop_preds
        kwargs["training_relevant_only"] = self.training_relevant_only
        return kwargs

    @property
    def include_targets(self) -> bool:
        return self.mode == self.MODE_TRAIN

    def empty_data(self) -> tuple[Data, torch.Tensor]:
        data = Data(
            x=torch.empty((0, self.NODE_FEATURE_DIM), dtype=torch.float32),
            edge_index=torch.empty((2, 0), dtype=torch.int64),
            edge_attr=torch.empty((0, self.EDGE_FEATURE_DIM), dtype=torch.float32),
            group_probs=torch.empty((0, self.NUM_CLASSES), dtype=torch.float32),
            splitter_probs=torch.empty((0, self.NUM_CLASSES), dtype=torch.float32),
            endpoint_preds=torch.empty((0, self.ENDPOINT_DIM), dtype=torch.float32),
            event_affinity=torch.empty((0, self.EVENT_AFFINITY_DIM), dtype=torch.float32),
            pion_stop_preds=torch.empty((0, self.PION_STOP_DIM), dtype=torch.float32),
            time_group_ids=torch.empty((0,), dtype=torch.int64),
        )
        targets = torch.empty((0, self.TARGET_DIM), dtype=torch.float32)
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
            and self.event_splitter_parquet_paths is None
            and self.pion_stop_parquet_paths is None
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
        pion_stop_cols = self.PION_STOP_COLUMNS

        for i in range(0, len(tasks), chunk_span):
            chunk_tasks = tasks[i : i + chunk_span]
            if not chunk_tasks:
                continue

            main_tables: list[pa.Table] = []
            group_prob_tables: list[pa.Table] = []
            splitter_prob_tables: list[pa.Table] = []
            endpoint_tables: list[pa.Table] = []
            event_splitter_tables: list[pa.Table] = []
            pion_stop_tables: list[pa.Table] = []

            for (
                main_path,
                group_prob_path,
                splitter_prob_path,
                endpoint_path,
                event_splitter_path,
                pion_stop_path,
                rg,
            ) in chunk_tasks:
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
                if event_splitter_path is not None:
                    es_pf = pq.ParquetFile(event_splitter_path)
                    event_raw = es_pf.read_row_group(rg, columns=self.EVENT_SPLITTER_COLUMNS)
                    event_splitter_tables.append(
                        self._rename_columns(event_raw, {"time_group_ids": "event_splitter_time_group_ids"})
                    )
                if pion_stop_path is not None:
                    ps_pf = pq.ParquetFile(pion_stop_path)
                    pion_raw = ps_pf.read_row_group(rg, columns=pion_stop_cols)
                    pion_stop_tables.append(
                        self._rename_columns(pion_raw, {"time_group_ids": "pion_stop_time_group_ids"})
                    )

            main_table = (
                main_tables[0] if len(main_tables) == 1 else pa.concat_tables(main_tables, promote_options="default")
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

            if event_splitter_tables:
                es_table = (
                    event_splitter_tables[0]
                    if len(event_splitter_tables) == 1
                    else pa.concat_tables(event_splitter_tables, promote_options="default")
                )
                if merged.num_rows != es_table.num_rows:
                    raise RuntimeError(
                        "Aligned chunk row mismatch between main and event_splitter tables: "
                        f"{merged.num_rows} vs {es_table.num_rows}"
                    )
                merged = self._merge_columns_from_table(merged, es_table, list(es_table.column_names))

            if pion_stop_tables:
                ps_table = (
                    pion_stop_tables[0]
                    if len(pion_stop_tables) == 1
                    else pa.concat_tables(pion_stop_tables, promote_options="default")
                )
                if merged.num_rows != ps_table.num_rows:
                    raise RuntimeError(
                        "Aligned chunk row mismatch between main and pion_stop tables: "
                        f"{merged.num_rows} vs {ps_table.num_rows}"
                    )
                merged = self._merge_columns_from_table(merged, ps_table, list(ps_table.column_names))

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
        tasks: list[tuple[str, str | None, str | None, str | None, str | None, str | None, int]],
    ) -> list[tuple[str, str | None, str | None, str | None, str | None, str | None, int]]:
        worker = get_worker_info()
        if worker is None:
            return tasks
        return tasks[worker.id :: worker.num_workers]

    def _aligned_row_group_tasks(
        self,
    ) -> list[tuple[str, str | None, str | None, str | None, str | None, str | None, int]]:
        tasks: list[tuple[str, str | None, str | None, str | None, str | None, str | None, int]] = []
        for i, main_path in enumerate(self.parquet_paths):
            group_path = None if self.group_probs_parquet_paths is None else self.group_probs_parquet_paths[i]
            splitter_path = None if self.group_splitter_parquet_paths is None else self.group_splitter_parquet_paths[i]
            endpoint_path = None if self.endpoint_parquet_paths is None else self.endpoint_parquet_paths[i]
            event_splitter_path = (
                None if self.event_splitter_parquet_paths is None else self.event_splitter_parquet_paths[i]
            )
            pion_stop_path = None if self.pion_stop_parquet_paths is None else self.pion_stop_parquet_paths[i]

            main_pf = pq.ParquetFile(main_path)
            gp_pf = pq.ParquetFile(group_path) if group_path is not None else None
            sp_pf = pq.ParquetFile(splitter_path) if splitter_path is not None else None
            ep_pf = pq.ParquetFile(endpoint_path) if endpoint_path is not None else None
            es_pf = pq.ParquetFile(event_splitter_path) if event_splitter_path is not None else None
            ps_pf = pq.ParquetFile(pion_stop_path) if pion_stop_path is not None else None

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
            if es_pf is not None and es_pf.num_row_groups != main_pf.num_row_groups:
                raise RuntimeError(
                    "Row-group mismatch between main and event_splitter files: "
                    f"{main_path} ({main_pf.num_row_groups}) vs {event_splitter_path} ({es_pf.num_row_groups})"
                )
            if ps_pf is not None and ps_pf.num_row_groups != main_pf.num_row_groups:
                raise RuntimeError(
                    "Row-group mismatch between main and pion_stop files: "
                    f"{main_path} ({main_pf.num_row_groups}) vs {pion_stop_path} ({ps_pf.num_row_groups})"
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
                            f"Row count mismatch at row-group {rg}: "
                            f"{main_path} ({n_main}) vs {splitter_path} ({n_sp})"
                        )
                if ep_pf is not None:
                    n_ep = int(ep_pf.metadata.row_group(rg).num_rows)
                    if n_ep != n_main:
                        raise RuntimeError(
                            f"Row count mismatch at row-group {rg}: "
                            f"{main_path} ({n_main}) vs {endpoint_path} ({n_ep})"
                        )
                if es_pf is not None:
                    n_es = int(es_pf.metadata.row_group(rg).num_rows)
                    if n_es != n_main:
                        raise RuntimeError(
                            f"Row count mismatch at row-group {rg}: "
                            f"{main_path} ({n_main}) vs {event_splitter_path} ({n_es})"
                        )
                if ps_pf is not None:
                    n_ps = int(ps_pf.metadata.row_group(rg).num_rows)
                    if n_ps != n_main:
                        raise RuntimeError(
                            f"Row count mismatch at row-group {rg}: "
                            f"{main_path} ({n_main}) vs {pion_stop_path} ({n_ps})"
                        )
                tasks.append((main_path, group_path, splitter_path, endpoint_path, event_splitter_path, pion_stop_path, rg))
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

        has_particle_mask = "hits_particle_mask" in table.column_names
        out["has_hits_particle_mask"] = np.asarray([1 if has_particle_mask else 0], dtype=np.int8)
        if has_particle_mask:
            pm_off, pm_vals = self._extract_list_column(table, "hits_particle_mask", np.int32)
            out["hits_particle_mask_offsets"] = pm_off
            out["hits_particle_mask_values"] = pm_vals
        has_pdg_id = "hits_pdg_id" in table.column_names
        out["has_hits_pdg_id"] = np.asarray([1 if has_pdg_id else 0], dtype=np.int8)
        if has_pdg_id:
            pdg_off, pdg_vals = self._extract_list_column(table, "hits_pdg_id", np.int32)
            out["hits_pdg_id_offsets"] = pdg_off
            out["hits_pdg_id_values"] = pdg_vals

        if self.include_targets:
            for col in self.TARGET_COLUMNS:
                off, vals = self._extract_list_column(table, col, np.float32)
                out[f"{col}_offsets"] = off
                out[f"{col}_values"] = vals

        has_group_prob_columns = all(col in table.column_names for col in self.GROUP_PROB_COLUMNS)
        out["has_group_prob_columns"] = np.asarray([1 if has_group_prob_columns else 0], dtype=np.int8)
        if has_group_prob_columns:
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

        has_splitter_prob_columns = all(col in table.column_names for col in self.SPLITTER_PROB_COLUMNS[:3])
        splitter_tg_col = None
        if "splitter_time_group_ids" in table.column_names:
            splitter_tg_col = "splitter_time_group_ids"
        elif "time_group_ids" in table.column_names:
            splitter_tg_col = "time_group_ids"
        out["has_splitter_prob_columns"] = np.asarray([1 if has_splitter_prob_columns else 0], dtype=np.int8)
        out["has_splitter_time_group_ids"] = np.asarray([1 if splitter_tg_col is not None else 0], dtype=np.int8)
        if has_splitter_prob_columns:
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
        elif "time_group_ids" in table.column_names:
            endpoint_tg_col = "time_group_ids"
        out["has_endpoint_quantile_columns"] = np.asarray([1 if has_endpoint_quant else 0], dtype=np.int8)
        out["has_endpoint_base_columns"] = np.asarray([1 if has_endpoint_base else 0], dtype=np.int8)
        out["has_endpoint_time_group_ids"] = np.asarray([1 if endpoint_tg_col is not None else 0], dtype=np.int8)
        if has_endpoint_quant:
            for col in endpoint_quant_cols:
                off, vals = self._extract_list_column(table, col, np.float32)
                out[f"{col}_offsets"] = off
                out[f"{col}_values"] = vals
        if has_endpoint_base:
            for col in self.ENDPOINT_BASE_COLUMNS:
                off, vals = self._extract_list_column(table, col, np.float32)
                out[f"{col}_offsets"] = off
                out[f"{col}_values"] = vals
        if endpoint_tg_col is not None:
            etg_off, etg_vals = self._extract_list_column(table, endpoint_tg_col, np.int64)
            out["endpoint_time_group_offsets"] = etg_off
            out["endpoint_time_group_values"] = etg_vals

        event_tg_col = None
        if "event_splitter_time_group_ids" in table.column_names:
            event_tg_col = "event_splitter_time_group_ids"
        elif (
            "time_group_ids" in table.column_names
            and "splitter_time_group_ids" not in table.column_names
            and "endpoint_time_group_ids" not in table.column_names
        ):
            event_tg_col = "time_group_ids"
        has_event_splitter_cols = all(
            col in table.column_names for col in ("edge_src_index", "edge_dst_index", "pred_edge_affinity")
        )
        out["has_event_splitter_columns"] = np.asarray(
            [1 if (event_tg_col is not None and has_event_splitter_cols) else 0],
            dtype=np.int8,
        )
        if event_tg_col is not None and has_event_splitter_cols:
            etg_off, etg_vals = self._extract_list_column(table, event_tg_col, np.int64)
            es_off, es_vals = self._extract_list_column(table, "edge_src_index", np.int64)
            ed_off, ed_vals = self._extract_list_column(table, "edge_dst_index", np.int64)
            ea_off, ea_vals = self._extract_list_column(table, "pred_edge_affinity", np.float32)
            out.update(
                {
                    "event_splitter_time_group_offsets": etg_off,
                    "event_splitter_time_group_values": etg_vals,
                    "event_edge_src_offsets": es_off,
                    "event_edge_src_values": es_vals,
                    "event_edge_dst_offsets": ed_off,
                    "event_edge_dst_values": ed_vals,
                    "event_edge_affinity_offsets": ea_off,
                    "event_edge_affinity_values": ea_vals,
                }
            )

        pion_tg_col = None
        if "pion_stop_time_group_ids" in table.column_names:
            pion_tg_col = "pion_stop_time_group_ids"
        elif "time_group_ids" in table.column_names:
            pion_tg_col = "time_group_ids"
        has_pion_stop_cols = all(col in table.column_names for col in self.PION_STOP_COLUMNS[:3])
        out["has_pion_stop_columns"] = np.asarray([1 if has_pion_stop_cols else 0], dtype=np.int8)
        out["has_pion_stop_time_group_ids"] = np.asarray([1 if pion_tg_col is not None else 0], dtype=np.int8)
        if has_pion_stop_cols:
            for col in self.PION_STOP_COLUMNS[:3]:
                off, vals = self._extract_list_column(table, col, np.float32)
                out[f"{col}_offsets"] = off
                out[f"{col}_values"] = vals
        if pion_tg_col is not None:
            ptg_off, ptg_vals = self._extract_list_column(table, pion_tg_col, np.int64)
            out["pion_stop_time_group_offsets"] = ptg_off
            out["pion_stop_time_group_values"] = ptg_vals

        return out

    @staticmethod
    def _particle_mask_to_multihot(mask_values: np.ndarray) -> np.ndarray:
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
                            raise RuntimeError("Splitter time_group_ids are not aligned with event hits.")
                        local_map[hit_pos] = s0 + split_pos
            out[h0:h1] = local_map
        return out

    def _fill_endpoint_preds_from_lists(
        self,
        *,
        endpoint_preds_out: np.ndarray,
        total_graphs: int,
        local_gid: np.ndarray,
        row_ids_graph: np.ndarray,
        chunk_in: dict[str, np.ndarray],
        row_group_counts: np.ndarray,
    ) -> None:
        has_quant = int(chunk_in["has_endpoint_quantile_columns"][0]) == 1
        has_base = int(chunk_in["has_endpoint_base_columns"][0]) == 1
        if not has_quant and not has_base:
            return

        def _validate_counts(offsets_name: str) -> None:
            counts = (chunk_in[offsets_name][1:] - chunk_in[offsets_name][:-1]).astype(np.int64, copy=False)
            if not np.array_equal(counts, row_group_counts):
                raise RuntimeError("Endpoint prediction list lengths do not match inferred row group counts.")

        endpoint_idx = 0
        for point in ("start", "end"):
            for coord in ("x", "y", "z"):
                base_name = f"pred_group_{point}_{coord}"
                if has_quant:
                    for q_idx, q in enumerate(self.ENDPOINT_QUANTILE_SUFFIXES):
                        col_name = f"{base_name}_{q}"
                        _validate_counts(f"{col_name}_offsets")
                        self._fill_graph_column_from_group_values(
                            out=endpoint_preds_out,
                            dst_col=endpoint_idx * self.NUM_TARGET_QUANTILES + q_idx,
                            vals=chunk_in[f"{col_name}_values"],
                            offs=chunk_in[f"{col_name}_offsets"],
                            total_graphs=total_graphs,
                            local_gid=local_gid,
                            row_ids_graph=row_ids_graph,
                        )
                elif has_base:
                    _validate_counts(f"{base_name}_offsets")
                    vals_tmp = np.zeros((total_graphs,), dtype=np.float32)
                    self._fill_graph_column_from_group_values(
                        out=vals_tmp.reshape(-1, 1),
                        dst_col=0,
                        vals=chunk_in[f"{base_name}_values"],
                        offs=chunk_in[f"{base_name}_offsets"],
                        total_graphs=total_graphs,
                        local_gid=local_gid,
                        row_ids_graph=row_ids_graph,
                    )
                    for q_idx in range(self.NUM_TARGET_QUANTILES):
                        endpoint_preds_out[:, endpoint_idx * self.NUM_TARGET_QUANTILES + q_idx] = vals_tmp
                endpoint_idx += 1

    @staticmethod
    def _fill_event_affinity_from_lists(
        *,
        event_affinity_out: np.ndarray,
        total_graphs: int,
        n_rows: int,
        row_group_counts: np.ndarray,
        row_group_base: np.ndarray,
        event_tg_offsets: np.ndarray,
        event_tg_values: np.ndarray,
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

            t0 = int(event_tg_offsets[row])
            t1 = int(event_tg_offsets[row + 1])
            if t1 > t0:
                tg_map = event_tg_values[t0:t1].astype(np.int64, copy=False)
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

    def _relevant_hit_mask(self, mask_values: np.ndarray) -> np.ndarray:
        # MIP-like hits: positron (4) or electron (8).
        return ((mask_values & 4) != 0) | ((mask_values & 8) != 0)

    def _compute_relevant_graph_mask(
        self,
        *,
        total_graphs: int,
        sorted_group_ids: np.ndarray,
        sorted_particle_mask: np.ndarray | None,
        sorted_pdg_id: np.ndarray | None = None,
    ) -> np.ndarray:
        if total_graphs <= 0:
            return np.zeros((0,), dtype=bool)
        if sorted_pdg_id is None:
            raise RuntimeError(
                f"{self.__class__.__name__} requires hits_pdg_id for training_relevant_only=True."
            )
        is_positron = sorted_pdg_id == int(self.POSITRON_PDG_ID)
        if is_positron.size == 0 or not np.any(is_positron):
            return np.zeros((total_graphs,), dtype=bool)
        counts = np.bincount(sorted_group_ids[is_positron], minlength=total_graphs).astype(np.int64, copy=False)
        return counts >= int(self.MIN_RELEVANT_HITS)

    @staticmethod
    def _select_graph_subset_pre_edges(
        *,
        keep_graph_ids: np.ndarray,
        node_ptr: np.ndarray,
        node_counts: np.ndarray,
        x: np.ndarray,
        time_group_ids: np.ndarray,
        splitter_probs: np.ndarray,
        coord_sorted: np.ndarray,
        z_sorted: np.ndarray,
        e_sorted: np.ndarray,
        view_sorted: np.ndarray,
        group_probs: np.ndarray,
        endpoint_preds: np.ndarray,
        event_affinity: np.ndarray,
        pion_stop_preds: np.ndarray,
        targets: np.ndarray | None,
        graph_event_ids: np.ndarray,
        graph_group_ids: np.ndarray,
    ) -> dict[str, np.ndarray]:
        keep = keep_graph_ids.astype(np.int64, copy=False)
        num_keep = int(keep.size)
        if num_keep == 0:
            return {
                "node_ptr": np.zeros((1,), dtype=np.int64),
                "node_counts": np.zeros((0,), dtype=np.int64),
                "x": np.zeros((0, x.shape[1]), dtype=x.dtype),
                "time_group_ids": np.zeros((0,), dtype=time_group_ids.dtype),
                "splitter_probs": np.zeros((0, splitter_probs.shape[1]), dtype=splitter_probs.dtype),
                "coord_sorted": np.zeros((0,), dtype=coord_sorted.dtype),
                "z_sorted": np.zeros((0,), dtype=z_sorted.dtype),
                "e_sorted": np.zeros((0,), dtype=e_sorted.dtype),
                "view_sorted": np.zeros((0,), dtype=view_sorted.dtype),
                "group_probs": np.zeros((0, group_probs.shape[1]), dtype=group_probs.dtype),
                "endpoint_preds": np.zeros((0, endpoint_preds.shape[1]), dtype=endpoint_preds.dtype),
                "event_affinity": np.zeros((0, event_affinity.shape[1]), dtype=event_affinity.dtype),
                "pion_stop_preds": np.zeros((0, pion_stop_preds.shape[1]), dtype=pion_stop_preds.dtype),
                "targets": (
                    np.zeros((0, targets.shape[1]), dtype=targets.dtype)
                    if targets is not None
                    else np.zeros((0, 0), dtype=np.float32)
                ),
                "graph_event_ids": np.zeros((0,), dtype=graph_event_ids.dtype),
                "graph_group_ids": np.zeros((0,), dtype=graph_group_ids.dtype),
            }

        keep_node_counts = node_counts[keep]
        new_node_ptr = np.zeros((num_keep + 1,), dtype=np.int64)
        new_node_ptr[1:] = np.cumsum(keep_node_counts, dtype=np.int64)
        total_nodes = int(new_node_ptr[-1])

        x_new = np.empty((total_nodes, x.shape[1]), dtype=x.dtype)
        tg_new = np.empty((total_nodes,), dtype=time_group_ids.dtype)
        splitter_new = np.empty((total_nodes, splitter_probs.shape[1]), dtype=splitter_probs.dtype)
        coord_new = np.empty((total_nodes,), dtype=coord_sorted.dtype)
        z_new = np.empty((total_nodes,), dtype=z_sorted.dtype)
        e_new = np.empty((total_nodes,), dtype=e_sorted.dtype)
        view_new = np.empty((total_nodes,), dtype=view_sorted.dtype)

        for new_gid, old_gid in enumerate(keep.tolist()):
            old_n0 = int(node_ptr[old_gid])
            old_n1 = int(node_ptr[old_gid + 1])
            new_n0 = int(new_node_ptr[new_gid])
            new_n1 = int(new_node_ptr[new_gid + 1])
            x_new[new_n0:new_n1] = x[old_n0:old_n1]
            tg_new[new_n0:new_n1] = time_group_ids[old_n0:old_n1]
            splitter_new[new_n0:new_n1] = splitter_probs[old_n0:old_n1]
            coord_new[new_n0:new_n1] = coord_sorted[old_n0:old_n1]
            z_new[new_n0:new_n1] = z_sorted[old_n0:old_n1]
            e_new[new_n0:new_n1] = e_sorted[old_n0:old_n1]
            view_new[new_n0:new_n1] = view_sorted[old_n0:old_n1]

        out: dict[str, np.ndarray] = {
            "node_ptr": new_node_ptr,
            "node_counts": keep_node_counts,
            "x": x_new,
            "time_group_ids": tg_new,
            "splitter_probs": splitter_new,
            "coord_sorted": coord_new,
            "z_sorted": z_new,
            "e_sorted": e_new,
            "view_sorted": view_new,
            "group_probs": group_probs[keep],
            "endpoint_preds": endpoint_preds[keep],
            "event_affinity": event_affinity[keep],
            "pion_stop_preds": pion_stop_preds[keep],
            "graph_event_ids": graph_event_ids[keep],
            "graph_group_ids": graph_group_ids[keep],
        }
        if targets is not None:
            out["targets"] = targets[keep]
        return out

    def _build_chunk_graph_arrays(self, *, table: pa.Table) -> dict:
        n_rows = int(table.num_rows)
        chunk_in = self._extract_chunk_inputs(table)
        row_group_count_candidates: list[np.ndarray] = []
        layout = self._compute_group_layout(
            n_rows=n_rows,
            hits_time_group_offsets=chunk_in["hits_time_group_offsets"],
            hits_time_group_values=chunk_in["hits_time_group_values"],
            row_group_count_candidates=row_group_count_candidates,
        )

        total_nodes = int(layout["total_nodes"])
        total_graphs = int(layout["total_graphs"])
        node_ptr = layout["node_ptr"]
        node_counts = layout["node_counts"]
        global_group_id = layout["global_group_id"]
        row_group_counts = layout["row_group_counts"]
        row_group_base = layout["row_group_base"]

        x_out = np.empty((total_nodes, self.NODE_FEATURE_DIM), dtype=np.float32)
        tgroup_out = np.empty((total_nodes,), dtype=np.int64)
        splitter_probs_out = np.zeros((total_nodes, self.NUM_CLASSES), dtype=np.float32)
        group_probs_out = np.zeros((total_graphs, self.NUM_CLASSES), dtype=np.float32)
        endpoint_preds_out = np.zeros((total_graphs, self.ENDPOINT_DIM), dtype=np.float32)
        event_affinity_out = np.zeros((total_graphs, self.EVENT_AFFINITY_DIM), dtype=np.float32)
        pion_stop_preds_out = np.zeros((total_graphs, self.PION_STOP_DIM), dtype=np.float32)
        y_base = np.zeros((total_graphs, self.TARGET_BASE_DIM), dtype=np.float32)

        row_ids_graph, local_gid, graph_event_ids, graph_group_ids = self._build_graph_index_mapping(
            total_graphs=total_graphs,
            n_rows=n_rows,
            row_group_counts=row_group_counts,
            row_group_base=row_group_base,
            event_ids=chunk_in["event_ids"],
        )

        node_populate_kwargs = {
            "total_nodes": total_nodes,
            "global_group_id": global_group_id,
            "hit_coord_values": chunk_in["hits_coord_values"],
            "hit_z_values": chunk_in["hits_z_values"],
            "hit_edep_values": chunk_in["hits_edep_values"],
            "hit_strip_type_values": chunk_in["hits_strip_type_values"],
            "hit_time_group_values": chunk_in["hits_time_group_values"],
            "x_out": x_out,
            "time_group_ids_out": tgroup_out,
        }
        if "hits_time_values" in chunk_in:
            node_populate_kwargs["hit_time_values"] = chunk_in["hits_time_values"]
        coord_sorted, z_sorted, e_sorted, view_sorted, order = self._populate_node_tensors(**node_populate_kwargs)

        node_truth = np.zeros((total_nodes, self.NUM_CLASSES), dtype=np.float32)
        group_truth = np.zeros((total_graphs, self.NUM_CLASSES), dtype=np.float32)
        sorted_gid = global_group_id[order] if total_nodes > 0 else np.zeros((0,), dtype=np.int64)
        sorted_mask: np.ndarray | None = None
        sorted_pdg: np.ndarray | None = None
        has_particle_mask = int(chunk_in["has_hits_particle_mask"][0]) == 1
        if has_particle_mask and total_nodes > 0:
            hit_counts = self._counts_from_offsets(chunk_in["hits_particle_mask_offsets"])
            if not np.array_equal(hit_counts, layout["hit_counts"]):
                raise RuntimeError("hits_particle_mask length mismatch with hit arrays in chunk.")
            sorted_mask = chunk_in["hits_particle_mask_values"][order]
            node_truth[:, :] = self._particle_mask_to_multihot(sorted_mask)
            for cls_id in range(self.NUM_CLASSES):
                cls_mask = node_truth[:, cls_id] > 0.5
                if not np.any(cls_mask):
                    continue
                present = np.bincount(sorted_gid[cls_mask], minlength=total_graphs) > 0
                group_truth[:, cls_id] = present.astype(np.float32, copy=False)
        has_pdg_id = int(chunk_in.get("has_hits_pdg_id", np.asarray([0], dtype=np.int8))[0]) == 1
        if has_pdg_id and total_nodes > 0:
            hit_counts = self._counts_from_offsets(chunk_in["hits_pdg_id_offsets"])
            if not np.array_equal(hit_counts, layout["hit_counts"]):
                raise RuntimeError("hits_pdg_id length mismatch with hit arrays in chunk.")
            sorted_pdg = chunk_in["hits_pdg_id_values"][order]

        has_prob_columns = int(chunk_in["has_group_prob_columns"][0]) == 1
        if self.use_group_probs:
            if has_prob_columns:
                pred_counts = (chunk_in["pred_pion_offsets"][1:] - chunk_in["pred_pion_offsets"][:-1]).astype(
                    np.int64,
                    copy=False,
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
            else:
                group_probs_out[:] = group_truth

        has_splitter_prob_columns = int(chunk_in["has_splitter_prob_columns"][0]) == 1
        if self.use_splitter_probs:
            if has_splitter_prob_columns:
                splitter_counts = (
                    chunk_in["pred_hit_pion_offsets"][1:] - chunk_in["pred_hit_pion_offsets"][:-1]
                ).astype(np.int64, copy=False)
                if not np.array_equal(splitter_counts, layout["hit_counts"]):
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
                idx_sorted = splitter_index_for_hit[order]
                splitter_probs_out[:, 0] = chunk_in["pred_hit_pion_values"][idx_sorted].astype(np.float32, copy=False)
                splitter_probs_out[:, 1] = chunk_in["pred_hit_muon_values"][idx_sorted].astype(np.float32, copy=False)
                splitter_probs_out[:, 2] = chunk_in["pred_hit_mip_values"][idx_sorted].astype(np.float32, copy=False)
            else:
                splitter_probs_out[:] = node_truth

        if self.use_endpoint_preds:
            self._fill_endpoint_preds_from_lists(
                endpoint_preds_out=endpoint_preds_out,
                total_graphs=total_graphs,
                local_gid=local_gid,
                row_ids_graph=row_ids_graph,
                chunk_in=chunk_in,
                row_group_counts=row_group_counts,
            )

        has_event_splitter_columns = int(chunk_in["has_event_splitter_columns"][0]) == 1
        if self.use_event_splitter_affinity and has_event_splitter_columns:
            self._fill_event_affinity_from_lists(
                event_affinity_out=event_affinity_out,
                total_graphs=total_graphs,
                n_rows=n_rows,
                row_group_counts=row_group_counts,
                row_group_base=row_group_base,
                event_tg_offsets=chunk_in["event_splitter_time_group_offsets"],
                event_tg_values=chunk_in["event_splitter_time_group_values"],
                edge_src_offsets=chunk_in["event_edge_src_offsets"],
                edge_src_values=chunk_in["event_edge_src_values"],
                edge_dst_offsets=chunk_in["event_edge_dst_offsets"],
                edge_dst_values=chunk_in["event_edge_dst_values"],
                edge_aff_offsets=chunk_in["event_edge_affinity_offsets"],
                edge_aff_values=chunk_in["event_edge_affinity_values"],
            )

        has_pion_stop_columns = int(chunk_in["has_pion_stop_columns"][0]) == 1
        if self.use_pion_stop_preds and has_pion_stop_columns:
            for i, col in enumerate(self.PION_STOP_COLUMNS[:3]):
                self._fill_graph_column_from_group_values(
                    out=pion_stop_preds_out,
                    dst_col=i,
                    vals=chunk_in[f"{col}_values"],
                    offs=chunk_in[f"{col}_offsets"],
                    total_graphs=total_graphs,
                    local_gid=local_gid,
                    row_ids_graph=row_ids_graph,
                )

        if self.include_targets:
            for col_idx, col_name in enumerate(self.TARGET_COLUMNS):
                counts = (chunk_in[f"{col_name}_offsets"][1:] - chunk_in[f"{col_name}_offsets"][:-1]).astype(
                    np.int64,
                    copy=False,
                )
                if not np.array_equal(counts, row_group_counts):
                    raise RuntimeError(f"{col_name} list lengths do not match inferred row group counts.")
                self._fill_graph_column_from_group_values(
                    out=y_base,
                    dst_col=col_idx,
                    vals=chunk_in[f"{col_name}_values"],
                    offs=chunk_in[f"{col_name}_offsets"],
                    total_graphs=total_graphs,
                    local_gid=local_gid,
                    row_ids_graph=row_ids_graph,
                )

        if self.include_targets and self.training_relevant_only:
            relevant_mask = self._compute_relevant_graph_mask(
                total_graphs=total_graphs,
                sorted_group_ids=sorted_gid,
                sorted_particle_mask=sorted_mask,
                sorted_pdg_id=sorted_pdg,
            )
            if relevant_mask.size == 0 or not np.any(relevant_mask):
                return {
                    "x": torch.empty((0, self.NODE_FEATURE_DIM), dtype=torch.float32),
                    "edge_index": torch.empty((2, 0), dtype=torch.int64),
                    "edge_attr": torch.empty((0, self.EDGE_FEATURE_DIM), dtype=torch.float32),
                    "group_probs": torch.empty((0, self.NUM_CLASSES), dtype=torch.float32),
                    "splitter_probs": torch.empty((0, self.NUM_CLASSES), dtype=torch.float32),
                    "endpoint_preds": torch.empty((0, self.ENDPOINT_DIM), dtype=torch.float32),
                    "event_affinity": torch.empty((0, self.EVENT_AFFINITY_DIM), dtype=torch.float32),
                    "pion_stop_preds": torch.empty((0, self.PION_STOP_DIM), dtype=torch.float32),
                    "time_group_ids": torch.empty((0,), dtype=torch.int64),
                    "graph_event_ids": torch.empty((0,), dtype=torch.int64),
                    "graph_group_ids": torch.empty((0,), dtype=torch.int64),
                    "node_ptr": torch.zeros((1,), dtype=torch.int64),
                    "edge_ptr": torch.zeros((1,), dtype=torch.int64),
                    "targets": torch.empty((0, self.TARGET_DIM), dtype=torch.float32),
                    "num_rows": n_rows,
                    "num_graphs": 0,
                }

            if not np.all(relevant_mask):
                subset = self._select_graph_subset_pre_edges(
                    keep_graph_ids=np.flatnonzero(relevant_mask),
                    node_ptr=node_ptr,
                    node_counts=node_counts,
                    x=x_out,
                    time_group_ids=tgroup_out,
                    splitter_probs=splitter_probs_out,
                    coord_sorted=coord_sorted,
                    z_sorted=z_sorted,
                    e_sorted=e_sorted,
                    view_sorted=view_sorted,
                    group_probs=group_probs_out,
                    endpoint_preds=endpoint_preds_out,
                    event_affinity=event_affinity_out,
                    pion_stop_preds=pion_stop_preds_out,
                    targets=y_base,
                    graph_event_ids=graph_event_ids,
                    graph_group_ids=graph_group_ids,
                )
                node_ptr = subset["node_ptr"]
                node_counts = subset["node_counts"]
                x_out = subset["x"]
                tgroup_out = subset["time_group_ids"]
                splitter_probs_out = subset["splitter_probs"]
                coord_sorted = subset["coord_sorted"]
                z_sorted = subset["z_sorted"]
                e_sorted = subset["e_sorted"]
                view_sorted = subset["view_sorted"]
                group_probs_out = subset["group_probs"]
                endpoint_preds_out = subset["endpoint_preds"]
                event_affinity_out = subset["event_affinity"]
                pion_stop_preds_out = subset["pion_stop_preds"]
                y_base = subset["targets"]
                graph_event_ids = subset["graph_event_ids"]
                graph_group_ids = subset["graph_group_ids"]
                total_graphs = int(group_probs_out.shape[0])
                total_nodes = int(x_out.shape[0])

        if self.include_targets:
            valid_target_mask = np.isfinite(y_base).all(axis=1)
            if valid_target_mask.size > 0 and not np.all(valid_target_mask):
                keep_ids = np.flatnonzero(valid_target_mask)
                if keep_ids.size == 0:
                    return {
                        "x": torch.empty((0, self.NODE_FEATURE_DIM), dtype=torch.float32),
                        "edge_index": torch.empty((2, 0), dtype=torch.int64),
                        "edge_attr": torch.empty((0, self.EDGE_FEATURE_DIM), dtype=torch.float32),
                        "group_probs": torch.empty((0, self.NUM_CLASSES), dtype=torch.float32),
                        "splitter_probs": torch.empty((0, self.NUM_CLASSES), dtype=torch.float32),
                        "endpoint_preds": torch.empty((0, self.ENDPOINT_DIM), dtype=torch.float32),
                        "event_affinity": torch.empty((0, self.EVENT_AFFINITY_DIM), dtype=torch.float32),
                        "pion_stop_preds": torch.empty((0, self.PION_STOP_DIM), dtype=torch.float32),
                        "time_group_ids": torch.empty((0,), dtype=torch.int64),
                        "graph_event_ids": torch.empty((0,), dtype=torch.int64),
                        "graph_group_ids": torch.empty((0,), dtype=torch.int64),
                        "node_ptr": torch.zeros((1,), dtype=torch.int64),
                        "edge_ptr": torch.zeros((1,), dtype=torch.int64),
                        "targets": torch.empty((0, self.TARGET_DIM), dtype=torch.float32),
                        "num_rows": n_rows,
                        "num_graphs": 0,
                    }
                subset = self._select_graph_subset_pre_edges(
                    keep_graph_ids=keep_ids,
                    node_ptr=node_ptr,
                    node_counts=node_counts,
                    x=x_out,
                    time_group_ids=tgroup_out,
                    splitter_probs=splitter_probs_out,
                    coord_sorted=coord_sorted,
                    z_sorted=z_sorted,
                    e_sorted=e_sorted,
                    view_sorted=view_sorted,
                    group_probs=group_probs_out,
                    endpoint_preds=endpoint_preds_out,
                    event_affinity=event_affinity_out,
                    pion_stop_preds=pion_stop_preds_out,
                    targets=y_base,
                    graph_event_ids=graph_event_ids,
                    graph_group_ids=graph_group_ids,
                )
                node_ptr = subset["node_ptr"]
                node_counts = subset["node_counts"]
                x_out = subset["x"]
                tgroup_out = subset["time_group_ids"]
                splitter_probs_out = subset["splitter_probs"]
                coord_sorted = subset["coord_sorted"]
                z_sorted = subset["z_sorted"]
                e_sorted = subset["e_sorted"]
                view_sorted = subset["view_sorted"]
                group_probs_out = subset["group_probs"]
                endpoint_preds_out = subset["endpoint_preds"]
                event_affinity_out = subset["event_affinity"]
                pion_stop_preds_out = subset["pion_stop_preds"]
                y_base = subset["targets"]
                graph_event_ids = subset["graph_event_ids"]
                graph_group_ids = subset["graph_group_ids"]
                total_graphs = int(group_probs_out.shape[0])
                total_nodes = int(x_out.shape[0])

        np.nan_to_num(x_out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(group_probs_out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(splitter_probs_out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(endpoint_preds_out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(event_affinity_out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(pion_stop_preds_out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if self.include_targets:
            np.nan_to_num(y_base, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        edge_counts = node_counts * np.maximum(node_counts - 1, 0)
        total_edges = int(edge_counts.sum())
        edge_ptr = np.zeros((total_graphs + 1,), dtype=np.int64)
        if total_graphs > 0:
            edge_ptr[1:] = np.cumsum(edge_counts, dtype=np.int64)

        edge_index_out = np.empty((2, total_edges), dtype=np.int64)
        edge_attr_out = np.empty((total_edges, self.EDGE_FEATURE_DIM), dtype=np.float32)
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
            "group_probs": torch.from_numpy(group_probs_out),
            "splitter_probs": torch.from_numpy(splitter_probs_out),
            "endpoint_preds": torch.from_numpy(endpoint_preds_out),
            "event_affinity": torch.from_numpy(event_affinity_out),
            "pion_stop_preds": torch.from_numpy(pion_stop_preds_out),
            "time_group_ids": torch.from_numpy(tgroup_out),
            "graph_event_ids": torch.from_numpy(graph_event_ids),
            "graph_group_ids": torch.from_numpy(graph_group_ids),
            "node_ptr": torch.from_numpy(node_ptr),
            "edge_ptr": torch.from_numpy(edge_ptr),
            "num_rows": n_rows,
            "num_graphs": int(total_graphs),
        }
        if self.include_targets:
            y_out = np.repeat(y_base, repeats=self.NUM_TARGET_QUANTILES, axis=1)
            chunk_out["targets"] = torch.from_numpy(y_out)
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
            group_probs=chunk["group_probs"][g0:g1],
            splitter_probs=chunk["splitter_probs"][n0:n1],
            endpoint_preds=chunk["endpoint_preds"][g0:g1],
            event_affinity=chunk["event_affinity"][g0:g1],
            pion_stop_preds=chunk["pion_stop_preds"][g0:g1],
            time_group_ids=chunk["time_group_ids"][n0:n1],
        )
        if "targets" in chunk:
            d.y = chunk["targets"][g0:g1]
        local_counts = (node_ptr[g0 + 1 : g1 + 1] - node_ptr[g0:g1]).to(dtype=torch.int64)
        d.batch = torch.repeat_interleave(torch.arange(g1 - g0, dtype=torch.int64), local_counts)
        d.event_ids = chunk["graph_event_ids"][g0:g1]
        d.group_ids = chunk["graph_group_ids"][g0:g1]
        d.num_graphs = int(g1 - g0)
        d.num_groups = int(g1 - g0)
        return d
