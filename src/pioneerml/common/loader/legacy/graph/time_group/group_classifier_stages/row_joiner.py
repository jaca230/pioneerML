from __future__ import annotations

import pyarrow as pa


class RowJoiner:
    """Row-level join stage for upstream tables.

    Group-classifier currently has no upstream join inputs, so this is a no-op.
    """

    def apply(self, *, table: pa.Table, loader) -> pa.Table:
        _ = loader
        return table
