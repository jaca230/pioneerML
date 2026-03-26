from __future__ import annotations

import pyarrow.parquet as pq


class ParquetSourceBackend:
    @staticmethod
    def schema_columns_intersection(main_sources: tuple[str, ...]) -> set[str]:
        if not main_sources:
            return set()
        intersection: set[str] | None = None
        for source in main_sources:
            names = set(pq.read_schema(source).names)
            intersection = names if intersection is None else (intersection & names)
            if not intersection:
                break
        return set(intersection or set())

