from __future__ import annotations

from pioneerml.common.plugins import NamespacedPluginRegistry

from ..base_evaluator import BaseEvaluator

REGISTRY = NamespacedPluginRegistry[type[BaseEvaluator]](
    namespace="evaluator",
    expected_type=BaseEvaluator,
    label="Evaluator plugin",
)
