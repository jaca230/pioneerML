
from zenml import step

from pioneerml.common.data_loader import LoaderFactory, BatchBundle
from pioneerml.common.pipeline.steps import BaseHPOStep, BaseLoaderStep

from ..objective import GroupClassifierObjectiveAdapter


class GroupClassifierHPOStep(BaseHPOStep):
    step_key = "hpo"

    def __init__(self, *, dataset: BatchBundle, pipeline_config: dict | None = None) -> None:
        super().__init__(pipeline_config=pipeline_config)
        self.dataset = dataset
        self.loader_factory = BaseLoaderStep.ensure_loader_factory(dataset, expected_type=LoaderFactory)
        self.objective_adapter = GroupClassifierObjectiveAdapter()

    def default_config(self) -> dict:
        return {
            "n_trials": 5,
            "max_epochs": 3,
            "grad_clip": 2.0,
            "trainer_kwargs": {"enable_progress_bar": True},
            "batch_size": {"min_exp": 5, "max_exp": 7},
            "shuffle": True,
            "chunk_row_groups": 4,
            "chunk_workers": None,
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
            "direction": "minimize",
            "seed": None,
            "study_name": "group_classifier_hpo",
            "storage": None,
            "fallback_dir": None,
            "allow_schema_fallback": True,
            "model": {
                "hidden": {"low": 64, "high": 256, "log": False},
                "heads": {"low": 2, "high": 8, "log": False},
                "num_blocks": {"low": 1, "high": 4, "log": False},
                "dropout": {"low": 0.0, "high": 0.3, "log": False},
            },
            "train": {
                "lr": {"low": 1e-4, "high": 1e-2, "log": True},
                "weight_decay": {"low": 1e-6, "high": 1e-3, "log": True},
                "scheduler_step_size": 2,
                "scheduler_gamma": 0.5,
                "threshold": 0.5,
            },
        }

    def objective_context(self) -> str:
        return "tune_group_classifier"

    def study_name_default(self) -> str:
        return "group_classifier_hpo"

    def build_hpo_loaders(self, *, cfg: dict, batch_size: int):
        train_params = BaseLoaderStep.resolve_loader_params(cfg, purpose="train", forced_batch_size=batch_size)
        val_params = BaseLoaderStep.resolve_loader_params(cfg, purpose="val", forced_batch_size=batch_size)
        train_provider = self.loader_factory.build_loader(loader_params=train_params)
        val_provider = self.loader_factory.build_loader(loader_params=val_params)
        if not train_provider.include_targets or not val_provider.include_targets:
            raise RuntimeError("GroupClassifierGraphLoader must run in train mode for HPO.")
        train_loader = train_provider.make_dataloader(shuffle_batches=bool(cfg.get("shuffle", True)))
        val_loader = val_provider.make_dataloader(shuffle_batches=False)
        return train_loader, val_loader


@step(name="tune_group_classifier")
def tune_group_classifier_step(
    dataset: BatchBundle,
    pipeline_config: dict | None = None,
) -> dict:
    return GroupClassifierHPOStep(dataset=dataset, pipeline_config=pipeline_config).execute()
