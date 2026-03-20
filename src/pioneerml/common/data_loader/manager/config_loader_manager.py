from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pioneerml.common.data_loader.loaders import DataFlowConfig, SplitSampleConfig

from .base_loader_manager import BaseLoaderManager
from .factory.registry import REGISTRY as LOADER_MANAGER_REGISTRY


@LOADER_MANAGER_REGISTRY.register("config")
class ConfigLoaderManager(BaseLoaderManager):
    @classmethod
    def from_factory(cls, *, config: Mapping[str, Any] | None = None) -> "ConfigLoaderManager":
        return cls(config=config)

    @staticmethod
    def _default_chunk_workers() -> int:
        import os

        cpu = int(os.cpu_count() or 1)
        return max(1, cpu - 1)

    def resolve_loader_params(
        self,
        *,
        purpose: str,
        forced_batch_size: int | None = None,
        default_batch_size: int = 64,
        default_chunk_row_groups: int = 4,
        default_mode: str = "train",
    ) -> dict[str, Any]:
        defaults_block_raw = self.config.get("defaults")
        if not isinstance(defaults_block_raw, Mapping):
            raise TypeError("config.defaults must be a mapping with keys ['type', 'config'].")
        defaults_block = dict(defaults_block_raw)
        default_type = defaults_block.get("type")
        if not isinstance(default_type, str) or default_type.strip() == "":
            raise TypeError("config.defaults.type must be a non-empty string.")
        defaults_cfg = defaults_block.get("config")
        if defaults_cfg is None:
            defaults_cfg = {}
        if not isinstance(defaults_cfg, Mapping):
            raise TypeError("config.defaults.config must be a mapping.")
        defaults = dict(defaults_cfg)

        purpose_key = self.loader_key_for_purpose(purpose=purpose)
        loaders_cfg = dict(self.loaders)
        purpose_block_raw = loaders_cfg.get(purpose_key)
        if purpose_block_raw is None:
            purpose_type = str(default_type).strip()
            purpose_cfg: dict[str, Any] = {}
        else:
            if not isinstance(purpose_block_raw, Mapping):
                raise TypeError(f"loaders.{purpose_key} must be a mapping with keys ['type', 'config'].")
            purpose_block = dict(purpose_block_raw)
            raw_type = purpose_block.get("type")
            purpose_type = str(default_type).strip() if raw_type is None else str(raw_type).strip()
            if purpose_type == "":
                raise TypeError(f"loaders.{purpose_key}.type must be a non-empty string.")
            raw_cfg = purpose_block.get("config")
            if raw_cfg is None:
                raw_cfg = {}
            if not isinstance(raw_cfg, Mapping):
                raise TypeError(f"loaders.{purpose_key}.config must be a mapping.")
            purpose_cfg = dict(raw_cfg)

        factory_type = str(self.loader_factory.plugin_name or "").strip()
        if factory_type != "" and purpose_type != factory_type:
            raise RuntimeError(
                f"config.loaders.{purpose_key}.type='{purpose_type}' does not match upstream loader_factory "
                f"plugin '{factory_type}'."
            )

        merged: dict[str, Any] = {**defaults, **purpose_cfg}

        if forced_batch_size is not None:
            merged["batch_size"] = int(forced_batch_size)
        else:
            raw_batch_size = merged.get("batch_size", default_batch_size)
            merged["batch_size"] = int(default_batch_size) if raw_batch_size is None else int(raw_batch_size)

        raw_chunk_row_groups = merged.get(
            "chunk_row_groups",
            merged.get("row_groups_per_chunk", default_chunk_row_groups),
        )
        merged["chunk_row_groups"] = max(1, int(raw_chunk_row_groups))

        chunk_workers = merged.get("chunk_workers", merged.get("num_workers"))
        if chunk_workers is None:
            chunk_workers = self._default_chunk_workers()
        merged["chunk_workers"] = max(0, int(chunk_workers))

        raw_mode = merged.get("mode", default_mode)
        merged["mode"] = str(default_mode) if raw_mode is None else str(raw_mode)

        split_seed_raw = merged.get("split_seed", None)
        split_seed = None if split_seed_raw in (None, "", "none", "None") else int(split_seed_raw)
        sample_fraction_raw = merged.get("sample_fraction")
        sample_fraction = None if sample_fraction_raw in (None, "", "none", "None") else float(sample_fraction_raw)
        split_raw = merged.get("split")
        split = None if split_raw in (None, "", "none", "None") else str(split_raw).strip().lower()

        merged["split_config"] = SplitSampleConfig(
            split=split,
            train_fraction=float(merged.get("train_fraction", 0.9)),
            val_fraction=float(merged.get("val_fraction", 0.05)),
            test_fraction=float(merged.get("test_fraction", 0.05)),
            split_seed=split_seed,
            sample_fraction=sample_fraction,
        )
        merged["data_flow_config"] = DataFlowConfig(
            batch_size=max(1, int(merged["batch_size"])),
            row_groups_per_chunk=max(1, int(merged["chunk_row_groups"])),
            num_workers=max(0, int(merged["chunk_workers"])),
        )
        return merged
