from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import get_worker_info


@dataclass(frozen=True)
class ParquetChunkReader:
    """Read parquet row groups in chunked table batches."""

    parquet_paths: list[str]
    columns: list[str]
    row_groups_per_chunk: int = 1

    def _row_group_tasks(self) -> list[tuple[str, int]]:
        tasks: list[tuple[str, int]] = []
        for path in self.parquet_paths:
            pf = pq.ParquetFile(path)
            for rg in range(pf.num_row_groups):
                tasks.append((path, rg))
        return tasks

    def _worker_shard(self, tasks: list[tuple[str, int]]) -> list[tuple[str, int]]:
        worker = get_worker_info()
        if worker is None:
            return tasks
        return tasks[worker.id :: worker.num_workers]

    def iter_tables(self) -> Iterator[pa.Table]:
        tasks = self._worker_shard(self._row_group_tasks())
        chunk_span = max(1, int(self.row_groups_per_chunk))
        for i in range(0, len(tasks), chunk_span):
            chunk_tasks = tasks[i : i + chunk_span]
            if not chunk_tasks:
                continue
            tables: list[pa.Table] = []
            for path, rg in chunk_tasks:
                pf = pq.ParquetFile(path)
                tables.append(pf.read_row_group(rg, columns=self.columns))
            table = tables[0] if len(tables) == 1 else pa.concat_tables(tables, promote_options="default")
            yield table.combine_chunks()
