from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from typing import Any

from .base_stage import BaseWriterStage


class ValidateInputsStage(BaseWriterStage):
    name = "validate_inputs"

    def __init__(self, *, required_state_keys: Sequence[str]) -> None:
        self.required_state_keys = tuple(str(k) for k in required_state_keys)

    def run_writer(self, *, state: MutableMapping[str, Any], owner) -> None:
        _ = owner
        missing = [k for k in self.required_state_keys if k not in state or state[k] is None]
        if missing:
            raise ValueError(f"Missing required writer inputs: {missing}")

