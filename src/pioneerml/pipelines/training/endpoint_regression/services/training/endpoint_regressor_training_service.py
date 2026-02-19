from __future__ import annotations

from pioneerml.common.loader import EndpointRegressorGraphLoader
from pioneerml.common.pipeline.services import BaseTrainingService
from pioneerml.common.pipeline.services.training.utils import GraphLightningModule

from ..base import EndpointRegressorServiceBase


class EndpointRegressorTrainingService(EndpointRegressorServiceBase, BaseTrainingService):
    step_key = "train"

    def __init__(
        self,
        *,
        dataset,
        pipeline_config: dict | None = None,
        hpo_params: dict | None = None,
    ) -> None:
        super().__init__(dataset=dataset, pipeline_config=pipeline_config)
        self.hpo_params = dict(hpo_params or {})

    def default_config(self) -> dict:
        return {
            "max_epochs": 10,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "grad_clip": 2.0,
            "scheduler_step_size": 2,
            "scheduler_gamma": 0.5,
            "trainer_kwargs": {"enable_progress_bar": True},
            "batch_size": 64,
            "shuffle": True,
            "chunk_row_groups": 4,
            "chunk_workers": 0,
            "use_group_probs": True,
            "use_splitter_probs": True,
            "early_stopping": {
                "enabled": False,
                "monitor": "val_loss",
                "mode": "min",
                "patience": 5,
                "min_delta": 0.0,
                "min_delta_mode": "absolute",
            },
            "compile": {"enabled": False, "mode": "default"},
            "model": {
                "in_channels": 4,
                "group_prob_dimension": 3,
                "splitter_prob_dimension": 3,
                "hidden": 192,
                "heads": 4,
                "layers": 3,
                "dropout": 0.1,
                "output_dim": int(EndpointRegressorGraphLoader.TARGET_DIM),
            },
        }

    def execute(self) -> GraphLightningModule:
        self.apply_warning_filter()
        cfg = self.get_config()
        if self.hpo_params:
            cfg = self._merge(cfg, self.hpo_params)

        model = self.objective_adapter.build_model(
            model_cfg=dict(cfg.get("model") or {}),
            compile_cfg=None,
            context="train_endpoint_regressor",
        )
        model = self.compile_model(model, compile_cfg=cfg.get("compile"), context="train_endpoint_regressor")
        module = self.objective_adapter.build_module(model=model, train_cfg=cfg)

        train_params = self._resolve_loader_params(cfg, purpose="train")
        val_params = self._resolve_loader_params(cfg, purpose="val")
        train_provider = self.loader_factory.build_loader(loader_params=train_params)
        val_provider = self.loader_factory.build_loader(loader_params=val_params)
        if not train_provider.include_targets or not val_provider.include_targets:
            raise RuntimeError("EndpointRegressorGraphLoader must run in train mode for training/validation.")
        train_loader = train_provider.make_dataloader(shuffle_batches=bool(cfg.get("shuffle", True)))
        val_loader = val_provider.make_dataloader(shuffle_batches=False)
        return self.fit_module(
            module=module,
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=int(cfg["max_epochs"]),
            grad_clip=float(cfg["grad_clip"]) if cfg.get("grad_clip") is not None else None,
            trainer_kwargs=dict(cfg.get("trainer_kwargs") or {}),
            early_stopping_cfg=dict(cfg.get("early_stopping") or {}),
        )
