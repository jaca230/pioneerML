
from typing import Any

from zenml import step

from pioneerml.common.loader import GroupClassifierGraphLoaderFactory, BatchBundle
from pioneerml.common.pipeline.steps import BaseLoaderStep, BaseTrainingStep
from pioneerml.common.pipeline.steps.training.utils import GraphLightningModule
from pioneerml.common.zenml.materializers import GraphLightningModuleMaterializer

from ..objective import GroupClassifierObjectiveAdapter


class GroupClassifierTrainStep(BaseTrainingStep):
    step_key = "train"

    def __init__(
        self,
        *,
        dataset: BatchBundle,
        pipeline_config: dict | None = None,
        hpo_params: dict | None = None,
    ) -> None:
        super().__init__(pipeline_config=pipeline_config)
        self.dataset = dataset
        self.hpo_params = dict(hpo_params or {})
        self.loader_factory = BaseLoaderStep.ensure_loader_factory(dataset, expected_type=GroupClassifierGraphLoaderFactory)
        self.objective_adapter = GroupClassifierObjectiveAdapter()

    def default_config(self) -> dict:
        return {
            "max_epochs": 10,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "grad_clip": 2.0,
            "scheduler_step_size": 2,
            "scheduler_gamma": 0.5,
            "threshold": 0.5,
            "trainer_kwargs": {"enable_progress_bar": True},
            "batch_size": 64,
            "shuffle": True,
            "chunk_row_groups": 4,
            "chunk_workers": None,
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
                "node_dim": 4,
                "edge_dim": 4,
                "graph_dim": 0,
                "hidden": 200,
                "heads": 4,
                "num_blocks": 2,
                "dropout": 0.1,
            },
        }

    def execute(self) -> GraphLightningModule:
        self.apply_warning_filter()
        cfg = self.get_config()
        if self.hpo_params:
            cfg = self.merge_config(cfg, self.hpo_params)

        model = self.objective_adapter.build_model(
            model_cfg=dict(cfg.get("model") or {}),
            compile_cfg=None,
            context="train_group_classifier",
        )
        model = self.compile_model(model, compile_cfg=cfg.get("compile"), context="train_group_classifier")
        module = self.objective_adapter.build_module(model=model, train_cfg=cfg)

        train_params = BaseLoaderStep.resolve_loader_params(cfg, purpose="train")
        val_params = BaseLoaderStep.resolve_loader_params(cfg, purpose="val")
        train_provider = self.loader_factory.build_loader(loader_params=train_params)
        val_provider = self.loader_factory.build_loader(loader_params=val_params)
        if not train_provider.include_targets or not val_provider.include_targets:
            raise RuntimeError("GroupClassifierGraphLoader must run in train mode for training/validation.")
        train_loader = train_provider.make_dataloader(shuffle_batches=bool(cfg.get("shuffle", True)))
        val_loader = val_provider.make_dataloader(shuffle_batches=False)
        module = self.fit_module(
            module=module,
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=int(cfg["max_epochs"]),
            grad_clip=float(cfg["grad_clip"]) if cfg.get("grad_clip") is not None else None,
            trainer_kwargs=dict(cfg.get("trainer_kwargs") or {}),
            early_stopping_cfg=dict(cfg.get("early_stopping") or {}),
        )
        BaseLoaderStep.log_loader_diagnostics(label="train", loader_provider=train_provider)
        BaseLoaderStep.log_loader_diagnostics(label="val", loader_provider=val_provider)
        return module


@step(name="train_group_classifier", output_materializers=GraphLightningModuleMaterializer)
def train_group_classifier_step(
    dataset: BatchBundle,
    pipeline_config: dict | None = None,
    hpo_params: dict | None = None,
) -> Any:
    return GroupClassifierTrainStep(dataset=dataset, pipeline_config=pipeline_config, hpo_params=hpo_params).execute()
