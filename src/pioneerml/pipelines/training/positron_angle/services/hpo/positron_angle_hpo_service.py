from __future__ import annotations

from pioneerml.common.pipeline.services import BaseHPOService

from ..base import PositronAngleServiceBase


class PositronAngleHPOService(PositronAngleServiceBase, BaseHPOService):
    step_key = "hpo"

    def __init__(self, *, dataset, pipeline_config: dict | None = None) -> None:
        super().__init__(dataset=dataset, pipeline_config=pipeline_config)

    def default_config(self) -> dict:
        return {
            "n_trials": 5,
            "max_epochs": 3,
            "grad_clip": 2.0,
            "trainer_kwargs": {"enable_progress_bar": True},
            "batch_size": {"min_exp": 5, "max_exp": 7},
            "shuffle": True,
            "chunk_row_groups": 4,
            "chunk_workers": 0,
            "use_group_probs": True,
            "use_splitter_probs": True,
            "use_endpoint_preds": True,
            "use_event_splitter_affinity": True,
            "training_relevant_only": True,
            "max_train_batches": None,
            "max_val_batches": None,
            "early_stopping": {
                "enabled": False,
                "monitor": "val_loss",
                "mode": "min",
                "patience": 3,
                "min_delta": 0.0,
                "min_delta_mode": "absolute",
            },
            "compile": {"enabled": False, "mode": "default"},
            "loss": {
                "pinball_weight": 0.0,
                "angular_weight": 1.0,
                "unit_norm_weight": 0.0,
                "normalize_target": False,
                "clamp_dot": False,
            },
            "direction": "minimize",
            "seed": None,
            "study_name": "positron_angle_hpo",
            "storage": None,
            "fallback_dir": None,
            "allow_schema_fallback": True,
            "model": {
                "hidden": {"low": 96, "high": 320, "log": False},
                "heads": {"low": 2, "high": 8, "log": False},
                "layers": {"low": 1, "high": 4, "log": False},
                "dropout": {"low": 0.0, "high": 0.3, "log": False},
            },
            "train": {
                "lr": {"low": 1e-4, "high": 1e-2, "log": True},
                "weight_decay": {"low": 1e-6, "high": 1e-3, "log": True},
                "scheduler_step_size": 2,
                "scheduler_gamma": 0.5,
            },
        }

    def objective_context(self) -> str:
        return "tune_positron_angle"

    def study_name_default(self) -> str:
        return "positron_angle_hpo"

    def build_hpo_loaders(self, *, cfg: dict, batch_size: int):
        train_params = self._resolve_loader_params(cfg, purpose="train", forced_batch_size=batch_size)
        val_params = self._resolve_loader_params(cfg, purpose="val", forced_batch_size=batch_size)
        train_provider = self.loader_factory.build_loader(loader_params=train_params)
        val_provider = self.loader_factory.build_loader(loader_params=val_params)
        if not train_provider.include_targets or not val_provider.include_targets:
            raise RuntimeError("PositronAngleGraphLoader must run in train mode for HPO.")
        train_loader = train_provider.make_dataloader(shuffle_batches=bool(cfg.get("shuffle", True)))
        val_loader = val_provider.make_dataloader(shuffle_batches=False)
        return train_loader, val_loader
