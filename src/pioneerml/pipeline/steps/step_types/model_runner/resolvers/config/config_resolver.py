from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pioneerml.data_loader.loaders.input_source import SourceType

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

        input_sources_spec = self._normalize_input_sources_spec(
            value=manager_cfg.get("input_sources_spec"),
            context="model_runner.loader_manager.config.input_sources_spec",
        )
        manager_cfg["input_sources_spec"] = input_sources_spec
        manager_cfg["input_backend"] = self._normalize_input_backend(
            value=manager_cfg.get("input_backend"),
        )

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

    @staticmethod
    def _normalize_input_backend(
        *,
        value: Any,
    ) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise TypeError(
                "model_runner.loader_manager.config.input_backend must be a mapping with keys ['type', 'config']."
            )
        block = dict(value)
        backend_type = block.get("type")
        if not isinstance(backend_type, str) or backend_type.strip() == "":
            raise TypeError("model_runner.loader_manager.config.input_backend.type must be a non-empty string.")
        backend_cfg = block.get("config")
        if backend_cfg is None:
            backend_cfg = {}
        if not isinstance(backend_cfg, Mapping):
            raise TypeError("model_runner.loader_manager.config.input_backend.config must be a mapping.")
        return {"type": str(backend_type).strip(), "config": dict(backend_cfg)}

    @staticmethod
    def _normalize_input_sources_spec(*, value: Any, context: str) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise TypeError(f"{context} must be a mapping.")
        spec = dict(value)
        main_sources = spec.get("main_sources")
        if main_sources is None:
            main_sources = spec.get("main_paths")
        optional_sources_by_name = spec.get("optional_sources_by_name")
        if optional_sources_by_name is None:
            optional_sources_by_name = spec.get("optional_paths_by_name")
        source_type = SourceType.from_value(spec.get("source_type", "file"))

        if not isinstance(main_sources, list):
            raise TypeError(f"{context}.main_sources (or main_paths) must be a list[str].")
        if optional_sources_by_name is None:
            optional_sources_by_name = {}
        if not isinstance(optional_sources_by_name, Mapping):
            raise TypeError(
                f"{context}.optional_sources_by_name (or optional_paths_by_name) must be a dict[str, list[str]|None]."
            )
        normalized_optional: dict[str, list[str] | None] = {}
        for key, values in dict(optional_sources_by_name).items():
            if not isinstance(key, str):
                raise TypeError(f"{context}.optional_sources_by_name keys must be strings.")
            if values is not None and not isinstance(values, list):
                raise TypeError(f"{context}.optional_sources_by_name values must be list[str] | None.")
            normalized_optional[str(key)] = (None if values is None else [str(v) for v in values])
        return {
            "main_sources": [str(v) for v in main_sources],
            "optional_sources_by_name": normalized_optional,
            "source_type": source_type.value,
        }
