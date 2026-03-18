from __future__ import annotations

from pioneerml.common.pipeline.steps import BaseEvaluationStep, BaseTrainingStep
from pioneerml.common.pipeline.steps.step_types.model_runner.training.hpo.utils import (
    with_trial_loader_split_seed,
)


class _DummyTrainingStep(BaseTrainingStep):
    step_key = "train"

    def _execute(self):
        return {}


class _DummyEvaluationStep(BaseEvaluationStep):
    step_key = "evaluate"

    def evaluate_from_loader(self, *, loader, cfg: dict, module, evaluator) -> dict:
        return {}


def test_model_runner_default_split_seed_is_static_for_training_and_evaluation():
    train = _DummyTrainingStep(pipeline_config={})
    evaluate = _DummyEvaluationStep(pipeline_config={})

    assert train.config_json["loader_config"]["base"]["split_seed"] == 0
    assert evaluate.config_json["loader_config"]["base"]["split_seed"] == 0


def test_hpo_fixed_mode_keeps_base_split_seed_when_no_seed_override():
    cfg = {
        "loader_split_seed_mode": "fixed",
        "loader_split_seed": None,
        "seed": None,
        "loader_config": {
            "base": {
                "split_seed": 0,
            }
        },
    }
    out = with_trial_loader_split_seed(cfg=cfg, trial_number=7)
    assert out["loader_config"]["base"]["split_seed"] == 0


def test_model_runner_shared_resolver_normalizes_split_seed_none_to_zero():
    train = _DummyTrainingStep(
        pipeline_config={"train": {"loader_config": {"base": {"split_seed": None}}}}
    )
    evaluate = _DummyEvaluationStep(
        pipeline_config={"evaluate": {"loader_config": {"base": {"split_seed": None}}}}
    )

    assert train.config_json["loader_config"]["base"]["split_seed"] == 0
    assert evaluate.config_json["loader_config"]["base"]["split_seed"] == 0
