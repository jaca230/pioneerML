from __future__ import annotations

from typing import Any

from ......resolver import BaseConfigResolver


class InferenceConfigResolver(BaseConfigResolver):
    def resolve(self, *, cfg: dict[str, Any]) -> None:
        cfg["threshold"] = float(cfg.get("threshold", 0.5))
        cfg["use_cuda"] = bool(cfg.get("use_cuda", True))
