from __future__ import annotations

from pioneerml.common.plugins import NamespacedPluginFactory

from ..base_evaluator import BaseEvaluator


class EvaluatorFactory(NamespacedPluginFactory[BaseEvaluator]):
    def __init__(
        self,
        *,
        evaluator_cls: type[BaseEvaluator] | None = None,
        evaluator_name: str | None = None,
    ) -> None:
        super().__init__(
            namespace="evaluator",
            plugin_cls=evaluator_cls,
            plugin_name=evaluator_name,
            expected_instance_type=BaseEvaluator,
            label="Evaluator",
        )
