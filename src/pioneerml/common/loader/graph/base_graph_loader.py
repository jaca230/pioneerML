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

    def _iter_batches(self, *, shuffle_batches: bool) -> Iterator[Data]:
        chunk_reader = ParquetChunkReader(
            parquet_paths=self.parquet_paths,
            columns=self.columns,
            row_groups_per_chunk=self.row_groups_per_chunk,
        )
        row_offset = 0
        for table in chunk_reader.iter_tables():
            chunk = self._build_chunk_graph_arrays(table=table)
            n_rows = int(chunk.get("num_rows", 0))
            if "graph_event_ids" in chunk and row_offset != 0:
                chunk["graph_event_ids"] = chunk["graph_event_ids"] + int(row_offset)
            num_graphs = int(chunk["num_graphs"])
            if num_graphs <= 0:
                row_offset += n_rows
                continue

            starts = torch.arange(0, num_graphs, self.batch_size, dtype=torch.int64)
            if shuffle_batches and starts.numel() > 1:
                starts = starts[torch.randperm(starts.numel())]
            for g0 in starts.tolist():
                g1 = min(g0 + self.batch_size, num_graphs)
                yield self._slice_chunk_batch(chunk, g0, g1)
            row_offset += n_rows

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
