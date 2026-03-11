from __future__ import annotations

import os
from collections.abc import MutableMapping
from typing import Any

import numpy as np
import pyarrow as pa

from ...utils.hashing import splitmix64
from .base_stage import BaseStage


class DistributedShardStage(BaseStage):
    """Deterministic row sharding across distributed ranks using event_id hashing."""

    name = "distributed_shard"
    requires = ("table",)
    provides = ("table",)

    def __init__(
        self,
        *,
        event_id_column: str = "event_id",
        enabled: bool = True,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> None:
        self.event_id_column = str(event_id_column)
        self.enabled = bool(enabled)
        self.rank = rank
        self.world_size = world_size

    @staticmethod
    def _from_env(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None:
            return int(default)
        try:
            return int(raw)
        except Exception:
            return int(default)

    def _resolve_rank_world(self) -> tuple[int, int]:
        if self.rank is not None and self.world_size is not None:
            return int(self.rank), int(self.world_size)

        try:
            import torch.distributed as dist  # type: ignore

            if dist.is_available() and dist.is_initialized():
                return int(dist.get_rank()), int(dist.get_world_size())
        except Exception:
            pass

        return (
            self._from_env("RANK", 0),
            self._from_env("WORLD_SIZE", 1),
        )

    def run_loader(self, *, state: MutableMapping[str, Any], owner) -> None:
        _ = owner
        if not self.enabled:
            return

        table = state.get("table")
        if table is None or int(table.num_rows) == 0:
            state["table"] = None
            state["chunk_out"] = None
            state["stop_pipeline"] = True
            return

        rank, world_size = self._resolve_rank_world()
        if int(world_size) <= 1:
            return
        if int(rank) < 0 or int(rank) >= int(world_size):
            raise RuntimeError(f"Invalid distributed rank/world_size: rank={rank}, world_size={world_size}")

        if self.event_id_column not in table.column_names:
            raise RuntimeError(f"{self.event_id_column} column is required for distributed sharding.")

        event_ids = table.column(self.event_id_column).chunk(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        event_u64 = event_ids.astype(np.uint64, copy=False)
        hashed = splitmix64(event_u64)
        mask = (hashed % np.uint64(int(world_size))) == np.uint64(int(rank))

        if mask.size == 0 or not np.any(mask):
            state["table"] = None
            state["chunk_out"] = None
            state["stop_pipeline"] = True
            return

        if not np.all(mask):
            table = table.filter(pa.array(mask))

        if int(table.num_rows) == 0:
            state["table"] = None
            state["chunk_out"] = None
            state["stop_pipeline"] = True
            return

        state["table"] = table
