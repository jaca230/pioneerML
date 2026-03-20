from __future__ import annotations

from pioneerml.common.plugins import NamespacedPluginFactory

from ..base_objective import BaseObjective


class ObjectiveFactory(NamespacedPluginFactory[BaseObjective]):
    def __init__(
        self,
        *,
        objective_cls: type[BaseObjective] | None = None,
        objective_name: str | None = None,
    ) -> None:
        super().__init__(
            namespace="objective",
            plugin_cls=objective_cls,
            plugin_name=objective_name,
            expected_instance_type=BaseObjective,
            label="Objective",
        )
