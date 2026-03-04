from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BaseStageContext:
    stage_index: int
    stage_name: str
