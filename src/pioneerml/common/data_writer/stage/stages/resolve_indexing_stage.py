from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from typing import Any

import numpy as np

from .base_stage import BaseWriterStage


class ResolveIndexingStage(BaseWriterStage):
    name = "resolve_indexing"

    def __init__(self, *, index_keys: Sequence[str]) -> None:
        self.index_keys = tuple(str(k) for k in index_keys)

    def run_writer(self, *, state: MutableMapping[str, Any], owner) -> None:
        _ = owner
        for key in self.index_keys:
            if key not in state or state[key] is None:
                continue
            state[key] = np.asarray(state[key], dtype=np.int64)

