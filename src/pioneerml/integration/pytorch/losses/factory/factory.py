from __future__ import annotations

from pioneerml.plugin import NamespacedPluginFactory

from ..base_loss import BaseLoss


class LossFactory(NamespacedPluginFactory[BaseLoss]):
    def __init__(
        self,
        *,
        loss_cls: type[BaseLoss] | None = None,
        loss_name: str | None = None,
    ) -> None:
        super().__init__(
            namespace="loss",
            plugin_cls=loss_cls,
            plugin_name=loss_name,
            expected_instance_type=BaseLoss,
            label="Loss",
        )

