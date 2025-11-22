"""
Base Stage class for pipeline framework.

Philosophy:
- Stages are arbitrary processing nodes (not prescriptive types)
- Stages declare inputs/outputs they work with
- Stages execute computations using shared context
- Stages can be connected in any DAG structure
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field


@dataclass
class StageConfig:
    """Configuration for a pipeline stage."""

    name: str
    """Unique name for this stage instance."""

    inputs: List[str] = field(default_factory=list)
    """List of context keys this stage reads from."""

    outputs: List[str] = field(default_factory=list)
    """List of context keys this stage writes to."""

    params: Dict[str, Any] = field(default_factory=dict)
    """Stage-specific parameters."""

    enabled: bool = True
    """Whether this stage is enabled."""


class Stage(ABC):
    """
    Base class for all pipeline stages.

    A Stage is a unit of computation that:
    - Reads data from a shared Context
    - Performs some operation
    - Writes results back to the Context

    Stages don't assume anything about what they process. They can be:
    - Data loaders
    - Preprocessors
    - Model trainers
    - Inference engines
    - Evaluators
    - Visualizers
    - Checkpointers
    - Custom transformations

    Example:
        >>> class MyStage(Stage):
        ...     def execute(self, context):
        ...         x = context['input_data']
        ...         result = process(x)
        ...         context['output_data'] = result
        ...
        >>> stage = MyStage(
        ...     config=StageConfig(
        ...         name='my_stage',
        ...         inputs=['input_data'],
        ...         outputs=['output_data'],
        ...     )
        ... )
    """

    def __init__(self, config: StageConfig):
        """
        Initialize stage with configuration.

        Args:
            config: Stage configuration including name, inputs, outputs, params.
        """
        self.config = config
        self._is_setup = False
        self._is_cleaned_up = False

    @property
    def name(self) -> str:
        """Get stage name."""
        return self.config.name

    @property
    def inputs(self) -> List[str]:
        """Get list of context keys this stage reads."""
        return self.config.inputs

    @property
    def outputs(self) -> List[str]:
        """Get list of context keys this stage writes."""
        return self.config.outputs

    @property
    def enabled(self) -> bool:
        """Check if stage is enabled."""
        return self.config.enabled

    def setup(self, context: Any) -> None:
        """
        Setup stage before execution.

        Called once before execute(). Use for:
        - Loading models/data
        - Initializing resources
        - Device allocation
        - One-time computations

        Args:
            context: Shared pipeline context.
        """
        self._is_setup = True

    @abstractmethod
    def execute(self, context: Any) -> None:
        """
        Execute the stage's computation.

        This is the main stage logic. Read from context using self.inputs,
        perform computation, write to context using self.outputs.

        Args:
            context: Shared pipeline context with read/write access.

        Raises:
            KeyError: If required input keys are missing from context.
        """
        pass

    def cleanup(self, context: Any) -> None:
        """
        Cleanup stage resources after execution.

        Called once after execute(). Use for:
        - Freeing memory
        - Closing files
        - Releasing GPU memory
        - Saving final state

        Args:
            context: Shared pipeline context.
        """
        self._is_cleaned_up = True

    def validate_inputs(self, context: Any) -> None:
        """
        Validate that required inputs exist in context.

        Args:
            context: Pipeline context to validate.

        Raises:
            KeyError: If required input key is missing.
        """
        missing = [key for key in self.inputs if key not in context]
        if missing:
            raise KeyError(f"Stage '{self.name}' missing required inputs: {missing}")

    def __repr__(self) -> str:
        """String representation of stage."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"inputs={self.inputs}, "
            f"outputs={self.outputs}, "
            f"enabled={self.enabled})"
        )


class FunctionalStage(Stage):
    """
    Stage that wraps a simple function.

    Useful for quick pipeline construction without defining full Stage classes.

    Example:
        >>> def my_function(context):
        ...     context['output'] = context['input'] * 2
        ...
        >>> stage = FunctionalStage(
        ...     config=StageConfig(
        ...         name='double',
        ...         inputs=['input'],
        ...         outputs=['output'],
        ...     ),
        ...     func=my_function,
        ... )
    """

    def __init__(self, config: StageConfig, func: callable):
        """
        Initialize functional stage.

        Args:
            config: Stage configuration.
            func: Function to execute. Should accept context as argument.
        """
        super().__init__(config)
        self.func = func

    def execute(self, context: Any) -> None:
        """Execute the wrapped function."""
        self.func(context)
