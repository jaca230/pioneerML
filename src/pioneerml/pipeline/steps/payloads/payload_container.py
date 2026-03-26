from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class StepPayloads(dict[str, Any]):
    """Container for upstream step payloads passed into step execution."""

    @classmethod
    def from_mapping(cls, payloads: Mapping[str, Any] | None) -> "StepPayloads":
        if payloads is None:
            return cls()
        return cls({str(k): v for k, v in dict(payloads).items()})
