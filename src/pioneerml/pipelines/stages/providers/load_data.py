"""
Stage for loading data into the pipeline context.
"""

from __future__ import annotations

from typing import Any

from pioneerml.pipelines.stage import StageConfig
from pioneerml.pipelines.stages.roles import ProviderStage


class LoadDataStage(ProviderStage):
    """
    Stage for loading data using a provided callable.

    Params:
        loader: Callable that returns the data object.
        output_key: Optional context key to store the data (default: first output or 'data').
    """

    def execute(self, context: Any) -> None:
        loader = self.config.params.get("loader")
        if loader is None:
            raise ValueError(f"Stage '{self.name}' requires 'loader' parameter")
        if not callable(loader):
            raise TypeError(f"loader must be callable, got {type(loader)}")

        data = loader()
        output_key = self.outputs[0] if self.outputs else "data"
        context[output_key] = data
