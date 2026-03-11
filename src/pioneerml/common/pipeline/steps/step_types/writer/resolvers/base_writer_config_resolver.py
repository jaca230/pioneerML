from __future__ import annotations

from typing import Any

from ....resolver import BaseConfigResolver


class BaseWriterConfigResolver(BaseConfigResolver):
    def resolve(self, *, cfg: dict[str, Any]) -> None:
        cfg["writer_config_json"] = dict(cfg.get("config_json") or {})
