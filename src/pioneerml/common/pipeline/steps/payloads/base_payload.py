from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class BaseStepPayload(dict[str, Any]):
    """Dict-like payload base with required-key validation and optional extra info."""

    REQUIRED_KEYS: tuple[str, ...] = ()

    def __init__(self, *args, extra_info: Mapping[str, Any] | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if extra_info is not None:
            self["extra_info"] = dict(extra_info)
        self.validate_required()

    def validate_required(self) -> None:
        missing = [key for key in self.REQUIRED_KEYS if key not in self or self.get(key) is None]
        if missing:
            raise RuntimeError(
                f"{self.__class__.__name__} missing required payload keys: {missing}"
            )

    def with_extra_info(self, **kwargs: Any) -> "BaseStepPayload":
        extra = dict(self.get("extra_info") or {})
        extra.update(dict(kwargs))
        self["extra_info"] = extra
        return self

