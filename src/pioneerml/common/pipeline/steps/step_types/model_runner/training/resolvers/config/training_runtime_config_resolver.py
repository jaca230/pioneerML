from __future__ import annotations

from typing import Any

from ......resolver import BaseConfigResolver


class TrainingRuntimeConfigResolver(BaseConfigResolver):
    def resolve(self, *, cfg: dict[str, Any]) -> None:
        self._validate_compile_cfg(cfg=cfg)
        self._validate_early_stopping_cfg(cfg=cfg)
        self._validate_loader_cfg(cfg=cfg)

    @staticmethod
    def _require_mapping(*, value: Any, context: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise TypeError(f"{context} must be a dict.")
        return dict(value)

    def _validate_compile_cfg(self, *, cfg: dict[str, Any]) -> None:
        compile_cfg = self._require_mapping(value=cfg.get("compile"), context="training.compile")
        required = (
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
        )
        missing = [k for k in required if k not in compile_cfg]
        if missing:
            raise KeyError(f"training.compile missing required keys: {missing}")

    def _validate_early_stopping_cfg(self, *, cfg: dict[str, Any]) -> None:
        es = self._require_mapping(value=cfg.get("early_stopping"), context="training.early_stopping")
        required_top = ("enabled", "type", "config")
        missing_top = [k for k in required_top if k not in es]
        if missing_top:
            raise KeyError(f"training.early_stopping missing required keys: {missing_top}")
        es_type = str(es["type"]).strip().lower()
        if es_type not in {"absolute", "default", "relative", "percent", "pct"}:
            raise ValueError(
                "training.early_stopping.type must be one of "
                "['absolute', 'default', 'relative', 'percent', 'pct']."
            )
        es_cfg = self._require_mapping(value=es.get("config"), context="training.early_stopping.config")
        required_cfg = ("monitor", "mode", "patience", "min_delta", "strict", "check_finite", "verbose")
        missing_cfg = [k for k in required_cfg if k not in es_cfg]
        if missing_cfg:
            raise KeyError(f"training.early_stopping.config missing required keys: {missing_cfg}")

    def _validate_loader_cfg(self, *, cfg: dict[str, Any]) -> None:
        loader_cfg = self._require_mapping(value=cfg.get("loader_config"), context="training.loader_config")
        required_top = ("base", "train", "val")
        missing_top = [k for k in required_top if k not in loader_cfg]
        if missing_top:
            raise KeyError(f"training.loader_config missing required keys: {missing_top}")
        base_cfg = self._require_mapping(value=loader_cfg.get("base"), context="training.loader_config.base")
        train_cfg = self._require_mapping(value=loader_cfg.get("train"), context="training.loader_config.train")
        val_cfg = self._require_mapping(value=loader_cfg.get("val"), context="training.loader_config.val")
        base_required = (
            "batch_size",
            "mode",
            "chunk_row_groups",
            "chunk_workers",
            "sample_fraction",
            "train_fraction",
            "val_fraction",
            "test_fraction",
            "split_seed",
        )
        base_missing = [k for k in base_required if k not in base_cfg]
        if base_missing:
            raise KeyError(f"training.loader_config.base missing required keys: {base_missing}")
        for name, split_cfg in (("train", train_cfg), ("val", val_cfg)):
            split_missing = [k for k in ("mode", "shuffle_batches", "log_diagnostics") if k not in split_cfg]
            if split_missing:
                raise KeyError(
                    f"training.loader_config.{name} missing required keys: {split_missing}"
                )
