from __future__ import annotations

import pyarrow as pa


class RowFilter:
    """Row-level pre-graph filter stage for group-classifier chunks."""

    def apply(self, *, table: pa.Table, loader) -> pa.Table:
        _ = loader
        return table
