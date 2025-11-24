"""
Stage for saving data from the pipeline context.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from pioneerml.pipelines.stage import StageConfig
from pioneerml.pipelines.stages.roles import ProviderStage


class SaveDataStage(ProviderStage):
    name = "save_data"
    """
    Stage for saving data using a provided saver or path.
    """

    def execute(self, context: Any) -> None:
        input_key = self.inputs[0] if self.inputs else "data"
        data = context[input_key]

        save_path = self.config.params.get("path")
        saver = self.config.params.get("saver")

        if saver is not None:
            saver(data, save_path)
        elif save_path is not None:
            np.save(save_path, data)
        else:
            raise ValueError(f"Stage '{self.name}' requires either 'path' or 'saver' parameter")
