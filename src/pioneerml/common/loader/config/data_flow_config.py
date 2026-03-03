from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataFlowConfig:
    """Execution-time data flow settings for loader chunking and mini-batching."""

    batch_size: int = 64
    row_groups_per_chunk: int = 4
    num_workers: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "batch_size", max(1, int(self.batch_size)))
        object.__setattr__(self, "row_groups_per_chunk", max(1, int(self.row_groups_per_chunk)))
        object.__setattr__(self, "num_workers", max(0, int(self.num_workers)))

