from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StageContext:
    chunk_index: int
    stage_index: int
    stage_name: str
    raw_num_rows: int
