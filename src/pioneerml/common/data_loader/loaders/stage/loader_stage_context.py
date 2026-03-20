from __future__ import annotations

from dataclasses import dataclass

from pioneerml.common.staged_runtime import BaseStageContext


@dataclass
class LoaderStageContext(BaseStageContext):
    chunk_index: int
    raw_num_rows: int
