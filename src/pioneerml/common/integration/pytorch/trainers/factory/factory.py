from __future__ import annotations

import pytorch_lightning as pl

from pioneerml.common.plugins import NamespacedPluginFactory


class TrainerFactory(NamespacedPluginFactory[pl.Trainer]):
    def __init__(
        self,
        *,
        trainer_cls: type[pl.Trainer] | None = None,
        trainer_name: str | None = None,
    ) -> None:
        super().__init__(
            namespace="trainer",
            plugin_cls=trainer_cls,
            plugin_name=trainer_name,
            expected_instance_type=pl.Trainer,
            label="Trainer",
        )
