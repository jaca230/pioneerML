from __future__ import annotations

from collections.abc import Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from pioneerml.common.integration.parquet import ParquetChunkReader, ParquetSourceBackend

from .base_backend import InputBackend


class ParquetInputBackend(InputBackend):
    """Parquet implementation of the input backend contract."""

    def schema_fields_intersection(self, sources: tuple[str, ...]) -> set[str]:
        return ParquetSourceBackend.schema_columns_intersection(sources)

    def iter_tables(
        self,
        *,
        sources: tuple[str, ...],
        fields: list[str],
        row_groups_per_chunk: int,
    ) -> Iterator[pa.Table]:
        reader = ParquetChunkReader(
            parquet_paths=list(sources),
            columns=list(fields),
            row_groups_per_chunk=max(1, int(row_groups_per_chunk)),
        )
        yield from reader.iter_tables()

    def count_rows_per_source(self, *, sources: tuple[str, ...]) -> list[int]:
        return [int(pq.ParquetFile(str(path)).metadata.num_rows) for path in sources]
