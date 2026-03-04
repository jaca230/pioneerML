from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .base_output_backend import OutputBackend


@dataclass
class _ParquetSink:
    dst_path: Path
    part_path: Path
    writer: pq.ParquetWriter | None = None


class ParquetOutputBackend(OutputBackend):
    def default_extension(self) -> str:
        return ".parquet"

    def write_table_atomic(self, *, table: pa.Table, dst_path: Path) -> None:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        part_path = dst_path.with_suffix(dst_path.suffix + ".part")
        if part_path.exists():
            part_path.unlink()
        pq.write_table(table, part_path)
        os.replace(part_path, dst_path)

    def open_sink(self, *, dst_path: Path) -> Any:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        part_path = dst_path.with_suffix(dst_path.suffix + ".part")
        if part_path.exists():
            part_path.unlink()
        if dst_path.exists():
            dst_path.unlink()
        return _ParquetSink(dst_path=dst_path, part_path=part_path, writer=None)

    def append_chunk(self, *, sink: Any, table: pa.Table) -> None:
        if not isinstance(sink, _ParquetSink):
            raise TypeError(f"Expected _ParquetSink, got {type(sink).__name__}.")
        if sink.writer is None:
            sink.writer = pq.ParquetWriter(str(sink.part_path), table.schema)
        sink.writer.write_table(table)

    def close_sink(self, *, sink: Any) -> None:
        if not isinstance(sink, _ParquetSink):
            raise TypeError(f"Expected _ParquetSink, got {type(sink).__name__}.")
        if sink.writer is not None:
            sink.writer.close()
            os.replace(sink.part_path, sink.dst_path)
