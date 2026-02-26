from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import DataLoader, IterableDataset
from torch_geometric.data import Data

from pioneerml.common.parquet import ParquetChunkReader


class BaseGraphLoader:
    """Base class for graph loaders with chunked minibatch iteration."""

    def __init__(
        self,
        parquet_paths: list[str],
        *,
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
        resolved = [str(p) for p in parquet_paths]
        if not resolved:
            raise RuntimeError("No parquet paths provided.")
        missing = [p for p in resolved if not Path(p).exists()]
        if missing:
            raise RuntimeError(f"Missing parquet path(s): {missing}")

        self.parquet_paths = resolved
        self.batch_size = max(1, int(batch_size))
        self.row_groups_per_chunk = max(1, int(row_groups_per_chunk))
        self.num_workers = max(0, int(num_workers))
        self.columns = list(columns or [])
        split_norm = None if split is None else str(split).strip().lower()
        if split_norm is not None and split_norm not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}. Expected one of: 'train', 'val', 'test'.")
        self.split = split_norm
        self.train_fraction = float(train_fraction)
        self.val_fraction = float(val_fraction)
        self.test_fraction = float(test_fraction)
        total_frac = self.train_fraction + self.val_fraction + self.test_fraction
        if self.split is not None and not np.isclose(total_frac, 1.0, atol=1e-9):
            raise ValueError(
                "train_fraction + val_fraction + test_fraction must sum to 1.0 "
                f"(got {self.train_fraction} + {self.val_fraction} + {self.test_fraction} = {total_frac})."
            )
        self.split_seed = int(split_seed)
        self.sample_fraction = None if sample_fraction is None else float(sample_fraction)
        if self.sample_fraction is not None and not (0.0 < self.sample_fraction <= 1.0):
            raise ValueError(f"sample_fraction must be in (0, 1], got: {self.sample_fraction}")
        self.edge_populate_graph_block = 512

    def with_runtime(
        self,
        *,
        batch_size: int | None = None,
        row_groups_per_chunk: int | None = None,
        num_workers: int | None = None,
    ):
        kwargs = self._constructor_kwargs()
        kwargs["batch_size"] = self.batch_size if batch_size is None else int(batch_size)
        kwargs["row_groups_per_chunk"] = (
            self.row_groups_per_chunk if row_groups_per_chunk is None else int(row_groups_per_chunk)
        )
        kwargs["num_workers"] = self.num_workers if num_workers is None else int(num_workers)
        return self.__class__(**kwargs)

    def _constructor_kwargs(self) -> dict[str, object]:
        return {
            "parquet_paths": self.parquet_paths,
            "batch_size": self.batch_size,
            "row_groups_per_chunk": self.row_groups_per_chunk,
            "num_workers": self.num_workers,
            "columns": self.columns,
            "split": self.split,
            "train_fraction": self.train_fraction,
            "val_fraction": self.val_fraction,
            "test_fraction": self.test_fraction,
            "split_seed": self.split_seed,
            "sample_fraction": self.sample_fraction,
        }

    def make_dataloader(self, *, shuffle_batches: bool) -> DataLoader:
        ds = _BaseGraphBatchIterable(self, shuffle_batches=bool(shuffle_batches))
        kwargs: dict[str, object] = {
            "batch_size": None,
            "num_workers": int(self.num_workers),
            "pin_memory": True,
        }
        if int(self.num_workers) > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 2
        return DataLoader(ds, **kwargs)

    def get_diagnostics_summary(self) -> dict:
        return {}

    def _iter_batches(self, *, shuffle_batches: bool) -> Iterator[Data]:
        chunk_reader = ParquetChunkReader(
            parquet_paths=self.parquet_paths,
            columns=self.columns,
            row_groups_per_chunk=self.row_groups_per_chunk,
        )
        row_offset = 0
        for table in chunk_reader.iter_tables():
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

    @staticmethod
    def _splitmix64(values: np.ndarray) -> np.ndarray:
        x = values.astype(np.uint64, copy=False)
        x = (x + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = x & np.uint64(0xFFFFFFFFFFFFFFFF)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x & np.uint64(0xFFFFFFFFFFFFFFFF)
        x = x ^ (x >> np.uint64(31))
        return x

    @classmethod
    def _uniform_from_hash(cls, hash_values: np.ndarray) -> np.ndarray:
        # Keep top 53 bits for deterministic float64 in [0, 1), matching IEEE mantissa precision.
        return ((hash_values >> np.uint64(11)).astype(np.float64)) * (1.0 / float(1 << 53))

    def _row_mask(self, event_ids: np.ndarray) -> np.ndarray:
        n_rows = int(event_ids.shape[0])
        if n_rows == 0:
            return np.zeros((0,), dtype=bool)

        event_u64 = event_ids.astype(np.uint64, copy=False)
        base = event_u64 ^ np.uint64(self.split_seed & 0xFFFFFFFFFFFFFFFF)
        u_split = self._uniform_from_hash(self._splitmix64(base))
        mask = np.ones((n_rows,), dtype=bool)

        if self.split is not None:
            train_hi = float(self.train_fraction)
            val_hi = train_hi + float(self.val_fraction)
            if self.split == "train":
                mask &= u_split < train_hi
            elif self.split == "val":
                mask &= (u_split >= train_hi) & (u_split < val_hi)
            elif self.split == "test":
                mask &= u_split >= val_hi

        if self.sample_fraction is not None:
            # Independent stream to avoid coupling sampling with split intervals.
            sample_seed = np.uint64((self.split_seed ^ 0xD6E8FEB86659FD93) & 0xFFFFFFFFFFFFFFFF)
            u_sample = self._uniform_from_hash(self._splitmix64(base ^ sample_seed))
            mask &= u_sample < float(self.sample_fraction)
        return mask

    def _filter_rows_before_graph_build(self, table: pa.Table) -> pa.Table | None:
        if self.split is None and self.sample_fraction is None:
            return table
        if table.num_rows == 0:
            return None
        if "event_id" not in table.column_names:
            raise RuntimeError("event_id column is required for split/sample filtering.")
        event_ids = self._to_np(table.column("event_id").chunk(0), np.int64)
        mask = self._row_mask(event_ids)
        if mask.size == 0 or not np.any(mask):
            return None
        if np.all(mask):
            return table
        return table.filter(pa.array(mask))

    def empty_data(self) -> tuple[Data, torch.Tensor]:
        raise NotImplementedError

    def _build_chunk_graph_arrays(self, *, table: pa.Table) -> dict:
        raise NotImplementedError

    @staticmethod
    def _slice_chunk_batch(chunk: dict, g0: int, g1: int) -> Data:
        raise NotImplementedError

    @staticmethod
    def _to_np(arr: pa.Array, dtype=None) -> np.ndarray:
        out = arr.to_numpy(zero_copy_only=False)
        if dtype is not None and out.dtype != dtype:
            out = out.astype(dtype, copy=False)
        return out

    @classmethod
    def _extract_list_column(cls, table: pa.Table, name: str, dtype) -> tuple[np.ndarray, np.ndarray]:
        arr = table.column(name).chunk(0)
        offsets = cls._to_np(arr.offsets, np.int64)
        values = cls._to_np(arr.values, dtype)
        return offsets, values

    @staticmethod
    def _coord_from_views(x: np.ndarray, y: np.ndarray, view: np.ndarray) -> np.ndarray:
        x_ok = np.where(np.isfinite(x), x, y)
        y_ok = np.where(np.isfinite(y), y, x)
        return np.where(view == 0, x_ok, y_ok).astype(np.float32, copy=False)

    @staticmethod
    def _complete_digraph_cached(
        k: int, cache: dict[int, tuple[np.ndarray, np.ndarray]]
    ) -> tuple[np.ndarray, np.ndarray]:
        tpl = cache.get(k)
        if tpl is not None:
            return tpl
        src = np.repeat(np.arange(k, dtype=np.int64), k)
        dst = np.tile(np.arange(k, dtype=np.int64), k)
        mask = src != dst
        tpl = (src[mask], dst[mask])
        cache[k] = tpl
        return tpl


class _BaseGraphBatchIterable(IterableDataset):
    def __init__(self, loader: BaseGraphLoader, *, shuffle_batches: bool) -> None:
        self._loader = loader
        self._shuffle_batches = bool(shuffle_batches)

    def __iter__(self):
        yield from self._loader._iter_batches(shuffle_batches=self._shuffle_batches)
