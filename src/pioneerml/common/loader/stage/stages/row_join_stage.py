from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import pyarrow as pa

from .base_stage import BaseStage


class RowJoinStage(BaseStage):
    """Base row-level join stage.

    Override `join_table` in subclasses to add upstream joins.
    """

    name = "row_join"
    requires = ("table",)
    provides = ("table",)

    def join_table(self, *, table: pa.Table, state: MutableMapping[str, Any], loader) -> pa.Table | None:
        _ = state
        _ = loader
        return table

    def run(self, *, state: MutableMapping[str, Any], loader) -> None:
        table = state.get("table")
        if table is None:
            state["chunk_out"] = None
            state["stop_pipeline"] = True
            return

        joined = self.join_table(table=table, state=state, loader=loader)
        if joined is None or int(joined.num_rows) == 0:
            state["table"] = None
            state["chunk_out"] = None
            state["stop_pipeline"] = True
            return
        state["table"] = joined
