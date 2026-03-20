from __future__ import annotations

from pioneerml.common.plugins import NamespacedPluginRegistry

from ..base_objective import BaseObjective

REGISTRY = NamespacedPluginRegistry[type[BaseObjective]](
    namespace="objective",
    expected_type=BaseObjective,
    label="Objective plugin",
)
