"""
Data-related pipeline stages.
"""

from __future__ import annotations

from typing import Any, Callable, Optional
from pathlib import Path

from pioneerml.pipelines.stage import Stage, StageConfig


class LoadDataStage(Stage):
    """
    Stage for loading data into the pipeline context.

    Example:
        >>> def load_fn():
        ...     return load_preprocessed_time_groups('/path/to/data/*.npy')
        ...
        >>> stage = LoadDataStage(
        ...     config=StageConfig(
        ...         name='load_data',
        ...         outputs=['dataset'],
        ...         params={'loader': load_fn},
        ...     )
        ... )
    """

    def execute(self, context: Any) -> None:
        """Load data using the provided loader function."""
        loader = self.config.params.get("loader")
        if loader is None:
            raise ValueError(f"Stage '{self.name}' requires 'loader' parameter")

        if not callable(loader):
            raise TypeError(f"loader must be callable, got {type(loader)}")

        # Load data
        data = loader()

        # Store in context
        output_key = self.outputs[0] if self.outputs else "data"
        context[output_key] = data


class SaveDataStage(Stage):
    """
    Stage for saving data from the pipeline context.

    Example:
        >>> stage = SaveDataStage(
        ...     config=StageConfig(
        ...         name='save_predictions',
        ...         inputs=['predictions'],
        ...         params={'path': 'outputs/predictions.npy'},
        ...     )
        ... )
    """

    def execute(self, context: Any) -> None:
        """Save data using the provided saver function or path."""
        import numpy as np

        input_key = self.inputs[0] if self.inputs else "data"
        data = context[input_key]

        # Get save path or custom saver
        save_path = self.config.params.get("path")
        saver = self.config.params.get("saver")

        if saver is not None:
            # Use custom saver function
            saver(data, save_path)
        elif save_path is not None:
            # Default: save as numpy
            np.save(save_path, data)
        else:
            raise ValueError(f"Stage '{self.name}' requires either 'path' or 'saver' parameter")
