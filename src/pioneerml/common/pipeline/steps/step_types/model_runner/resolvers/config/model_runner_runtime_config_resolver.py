from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .....resolver import BaseConfigResolver


class ModelRunnerRuntimeConfigResolver(BaseConfigResolver):
    def resolve(self, *, cfg: dict[str, Any]) -> None:
        self._normalize_compile_cfg(cfg=cfg)
        self._normalize_loader_cfg(cfg=cfg)

    @staticmethod
    def _normalize_compile_cfg(*, cfg: dict[str, Any]) -> None:
        raw = cfg.get("compile")
        if raw is None:
            cfg["compile"] = {}
            return
        if not isinstance(raw, Mapping):
            raise TypeError("model_runner.compile must be a mapping when provided.")
        cfg["compile"] = dict(raw)

    @staticmethod
    def _normalize_loader_cfg(*, cfg: dict[str, Any]) -> None:
        raw_loader_cfg = cfg.get("loader_config")
        if raw_loader_cfg is None:
            cfg["loader_config"] = {"base": {}}
            raw_loader_cfg = cfg["loader_config"]
        if not isinstance(raw_loader_cfg, Mapping):
            raise TypeError("model_runner.loader_config must be a mapping when provided.")

        loader_cfg = dict(raw_loader_cfg)
        base_cfg = loader_cfg.get("base")
        if base_cfg is None:
            base_cfg = {}
        if not isinstance(base_cfg, Mapping):
            raise TypeError("model_runner.loader_config.base must be a mapping when provided.")
        base = dict(base_cfg)

        # Keep shared data split deterministic across all loader uses by default.
        if base.get("split_seed") is None:
            base["split_seed"] = 0

        loader_cfg["base"] = base
        cfg["loader_config"] = loader_cfg
