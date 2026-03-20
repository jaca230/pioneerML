from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .....resolver import BaseConfigResolver


class ModelRunnerConfigResolver(BaseConfigResolver):
    def resolve(self, *, cfg: dict[str, Any]) -> None:
        self._normalize_plugin_cfg(
            cfg=cfg,
            key="compiler",
            required_config_keys=(
                "enabled",
                "mode",
                "dynamic",
                "backend",
                "matmul_precision",
                "capture_scalar_outputs",
                "cudagraph_skip_dynamic_graphs",
                "max_autotune",
                "max_autotune_gemm",
                "inductor_log_level",
            ),
        )
        self._normalize_loader_manager_cfg(cfg=cfg)

    @staticmethod
    def _normalize_plugin_cfg(
        *,
        cfg: dict[str, Any],
        key: str,
        required_config_keys: tuple[str, ...] = (),
    ) -> None:
        raw = cfg.get(key)
        if not isinstance(raw, Mapping):
            raise TypeError(f"model_runner.{key} must be a mapping with keys: ['type', 'config'].")
        block = dict(raw)
        plugin_type = block.get("type")
        if not isinstance(plugin_type, str) or plugin_type.strip() == "":
            raise ValueError(f"model_runner.{key}.type must be a non-empty string.")
        plugin_cfg = block.get("config")
        if not isinstance(plugin_cfg, Mapping):
            raise TypeError(f"model_runner.{key}.config must be a mapping.")
        plugin_cfg = dict(plugin_cfg)
        missing = [k for k in required_config_keys if k not in plugin_cfg]
        if missing:
            raise KeyError(f"model_runner.{key}.config missing required keys: {missing}")
        cfg[key] = {"type": str(plugin_type).strip(), "config": plugin_cfg}

    def _normalize_loader_manager_cfg(self, *, cfg: dict[str, Any]) -> None:
        self._normalize_plugin_cfg(cfg=cfg, key="loader_manager")
        manager_block = dict(cfg.get("loader_manager") or {})
        manager_cfg = manager_block.get("config")
        if not isinstance(manager_cfg, Mapping):
            raise TypeError("model_runner.loader_manager.config must be a mapping.")
        manager_cfg = dict(manager_cfg)

        defaults_block = manager_cfg.get("defaults")
        if not isinstance(defaults_block, Mapping):
            raise TypeError("model_runner.loader_manager.config.defaults must be a mapping with keys ['type', 'config'].")
        defaults_block = dict(defaults_block)
        default_loader_type = defaults_block.get("type")
        if not isinstance(default_loader_type, str) or default_loader_type.strip() == "":
            raise ValueError("model_runner.loader_manager.config.defaults.type must be a non-empty string.")
        default_loader_cfg = defaults_block.get("config")
        if default_loader_cfg is None:
            default_loader_cfg = {}
        if not isinstance(default_loader_cfg, Mapping):
            raise TypeError("model_runner.loader_manager.config.defaults.config must be a mapping.")
        default_loader_cfg = dict(default_loader_cfg)

        loaders_cfg = manager_cfg.get("loaders")
        if not isinstance(loaders_cfg, Mapping):
            raise TypeError("model_runner.loader_manager.config.loaders must be a mapping.")
        loaders_cfg = dict(loaders_cfg)

        normalized_loaders: dict[str, dict[str, Any]] = {}
        for loader_key, raw_loader in loaders_cfg.items():
            if not isinstance(raw_loader, Mapping):
                raise TypeError(
                    f"model_runner.loader_manager.config.loaders.{loader_key} must be a mapping with keys ['type', 'config']."
                )
            loader_block = dict(raw_loader)
            loader_type_raw = loader_block.get("type")
            loader_type = str(default_loader_type).strip() if loader_type_raw is None else str(loader_type_raw).strip()
            if loader_type == "":
                raise ValueError(f"model_runner.loader_manager.config.loaders.{loader_key}.type must be a non-empty string.")
            loader_cfg = loader_block.get("config")
            if loader_cfg is None:
                loader_cfg = {}
            if not isinstance(loader_cfg, Mapping):
                raise TypeError(f"model_runner.loader_manager.config.loaders.{loader_key}.config must be a mapping.")
            normalized_loaders[str(loader_key)] = {"type": loader_type, "config": dict(loader_cfg)}

        # Keep shared split deterministic by default and keep batch_size optional for HPO control.
        if default_loader_cfg.get("split_seed") is None:
            default_loader_cfg["split_seed"] = 0
        if "batch_size" not in default_loader_cfg:
            default_loader_cfg["batch_size"] = None

        manager_cfg["defaults"] = {"type": str(default_loader_type).strip(), "config": default_loader_cfg}
        manager_cfg["loaders"] = normalized_loaders
        manager_block["config"] = manager_cfg
        cfg["loader_manager"] = manager_block
