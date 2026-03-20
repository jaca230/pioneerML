from __future__ import annotations

from pioneerml.common.plugins import NamespacedPluginFactory

from ..base_hpo import BaseHPO


class HPOFactory(NamespacedPluginFactory[BaseHPO]):
    def __init__(
        self,
        *,
        hpo_cls: type[BaseHPO] | None = None,
        hpo_name: str | None = None,
    ) -> None:
        super().__init__(
            namespace="hpo",
            plugin_cls=hpo_cls,
            plugin_name=hpo_name,
            expected_instance_type=BaseHPO,
            label="HPO",
        )
