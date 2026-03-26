from __future__ import annotations

from typing import Any

from ......resolver import BaseConfigResolver


class TrainingConfigResolver(BaseConfigResolver):
    def resolve(self, *, cfg: dict[str, Any]) -> None:
        self._validate_architecture_cfg(cfg=cfg)
        self._validate_module_cfg(cfg=cfg)
        self._validate_trainer_cfg(cfg=cfg)
        self._validate_loader_cfg(cfg=cfg)
        self._validate_log_filter_cfg(cfg=cfg)

    @staticmethod
    def _require_mapping(*, value: Any, context: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise TypeError(f"{context} must be a dict.")
        return dict(value)

    def _validate_architecture_cfg(self, *, cfg: dict[str, Any]) -> None:
        architecture = self._require_mapping(value=cfg.get("architecture"), context="training.architecture")
        arch_type = architecture.get("type")
        if not isinstance(arch_type, str) or arch_type.strip() == "":
            raise TypeError("training.architecture.type must be a non-empty string.")
        if str(arch_type).strip().lower() == "required":
            raise ValueError("training.architecture.type must be set to a concrete registered architecture plugin.")
        arch_config = architecture.get("config")
        if not isinstance(arch_config, dict):
            raise TypeError("training.architecture.config must be a dict.")

    def _validate_module_cfg(self, *, cfg: dict[str, Any]) -> None:
        module = self._require_mapping(value=cfg.get("module"), context="training.module")
        module_type = module.get("type")
        if not isinstance(module_type, str) or module_type.strip() == "":
            raise TypeError("training.module.type must be a non-empty string.")
        module_config = module.get("config")
        if not isinstance(module_config, dict):
            raise TypeError("training.module.config must be a dict.")

    def _validate_trainer_cfg(self, *, cfg: dict[str, Any]) -> None:
        trainer = self._require_mapping(value=cfg.get("trainer"), context="training.trainer")
        trainer_type = trainer.get("type")
        if not isinstance(trainer_type, str) or trainer_type.strip() == "":
            raise TypeError("training.trainer.type must be a non-empty string.")
        trainer_config = self._require_mapping(value=trainer.get("config"), context="training.trainer.config")

        required = ("trainer_kwargs", "early_stopping")
        missing = [k for k in required if k not in trainer_config]
        if missing:
            raise KeyError(f"training.trainer.config missing required keys: {missing}")

        trainer_kwargs = trainer_config.get("trainer_kwargs")
        if not isinstance(trainer_kwargs, dict):
            raise TypeError("training.trainer.config.trainer_kwargs must be a dict.")

        es = self._require_mapping(
            value=trainer_config.get("early_stopping"),
            context="training.trainer.config.early_stopping",
        )
        required_top = ("enabled", "type", "config")
        missing_top = [k for k in required_top if k not in es]
        if missing_top:
            raise KeyError(f"training.trainer.config.early_stopping missing required keys: {missing_top}")
        es_type = str(es["type"]).strip().lower()
        if es_type not in {"absolute", "default", "relative", "percent", "pct"}:
            raise ValueError(
                "training.trainer.config.early_stopping.type must be one of "
                "['absolute', 'default', 'relative', 'percent', 'pct']."
            )
        es_cfg = self._require_mapping(
            value=es.get("config"),
            context="training.trainer.config.early_stopping.config",
        )
        required_cfg = ("monitor", "mode", "patience", "min_delta", "strict", "check_finite", "verbose")
        missing_cfg = [k for k in required_cfg if k not in es_cfg]
        if missing_cfg:
            raise KeyError(f"training.trainer.config.early_stopping.config missing required keys: {missing_cfg}")

    def _validate_loader_cfg(self, *, cfg: dict[str, Any]) -> None:
        loader_manager = self._require_mapping(value=cfg.get("loader_manager"), context="training.loader_manager")
        loader_manager_cfg = self._require_mapping(
            value=loader_manager.get("config"),
            context="training.loader_manager.config",
        )
        defaults = self._require_mapping(
            value=loader_manager_cfg.get("defaults"),
            context="training.loader_manager.config.defaults",
        )
        defaults_type = defaults.get("type")
        if not isinstance(defaults_type, str) or defaults_type.strip() == "":
            raise TypeError("training.loader_manager.config.defaults.type must be a non-empty string.")
        defaults_cfg = self._require_mapping(
            value=defaults.get("config"),
            context="training.loader_manager.config.defaults.config",
        )
        loader_cfg = self._require_mapping(
            value=loader_manager_cfg.get("loaders"),
            context="training.loader_manager.config.loaders",
        )

        required_top = ("train_loader", "val_loader")
        missing_top = [k for k in required_top if k not in loader_cfg]
        if missing_top:
            raise KeyError(f"training.loader_manager.config.loaders missing required keys: {missing_top}")
        train_block = self._require_mapping(
            value=loader_cfg.get("train_loader"),
            context="training.loader_manager.config.loaders.train_loader",
        )
        val_block = self._require_mapping(
            value=loader_cfg.get("val_loader"),
            context="training.loader_manager.config.loaders.val_loader",
        )
        train_type = train_block.get("type")
        if train_type is not None and (not isinstance(train_type, str) or train_type.strip() == ""):
            raise TypeError(
                "training.loader_manager.config.loaders.train_loader.type must be a non-empty string when provided."
            )
        val_type = val_block.get("type")
        if val_type is not None and (not isinstance(val_type, str) or val_type.strip() == ""):
            raise TypeError(
                "training.loader_manager.config.loaders.val_loader.type must be a non-empty string when provided."
            )
        train_cfg = self._require_mapping(
            value=train_block.get("config"),
            context="training.loader_manager.config.loaders.train_loader.config",
        )
        val_cfg = self._require_mapping(
            value=val_block.get("config"),
            context="training.loader_manager.config.loaders.val_loader.config",
        )

        base_required = (
            "mode",
            "chunk_row_groups",
            "chunk_workers",
            "sample_fraction",
            "train_fraction",
            "val_fraction",
            "test_fraction",
            "split_seed",
        )
        base_missing = [k for k in base_required if k not in defaults_cfg]
        if base_missing:
            raise KeyError(f"training.loader_manager.config.defaults.config missing required keys: {base_missing}")
        for name, split_cfg in (("train_loader", train_cfg), ("val_loader", val_cfg)):
            split_missing = [k for k in ("mode", "shuffle_batches", "log_diagnostics") if k not in split_cfg]
            if split_missing:
                raise KeyError(
                    f"training.loader_manager.config.loaders.{name}.config missing required keys: {split_missing}"
                )

    def _validate_log_filter_cfg(self, *, cfg: dict[str, Any]) -> None:
        log_filter = self._require_mapping(value=cfg.get("log_filter"), context="training.log_filter")
        raw_name = log_filter.get("type")
        if not isinstance(raw_name, str) or raw_name.strip() == "":
            raise TypeError("training.log_filter.type must be a non-empty string.")
        raw_cfg = log_filter.get("config")
        if raw_cfg is None:
            log_filter["config"] = {}
            cfg["log_filter"] = log_filter
            return
        if not isinstance(raw_cfg, dict):
            raise TypeError("training.log_filter.config must be a dict.")
