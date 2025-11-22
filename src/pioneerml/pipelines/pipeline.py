"""
Pipeline executor with DAG-based execution.

Pipelines execute stages in topological order based on dependencies.
"""

from __future__ import annotations

from typing import List, Dict, Set, Optional, Any
from collections import deque, defaultdict
import logging

from pioneerml.pipelines.stage import Stage
from pioneerml.pipelines.context import Context


logger = logging.getLogger(__name__)


class Pipeline:
    """
    DAG-based pipeline executor.

    A Pipeline manages the execution of multiple stages in dependency order.
    Dependencies are automatically inferred from stage inputs/outputs.

    Features:
    - Automatic dependency resolution via topological sort
    - Parallel execution of independent stages (future)
    - Resume from checkpoint
    - Skip completed stages
    - Validation of stage connectivity

    Example:
        >>> from pioneerml.pipelines import Pipeline, Context
        >>> from myapp.stages import LoadData, Train, Evaluate
        >>>
        >>> pipeline = Pipeline([
        ...     LoadData(name='load', outputs=['dataset']),
        ...     Train(name='train', inputs=['dataset'], outputs=['model']),
        ...     Evaluate(name='eval', inputs=['model', 'dataset'], outputs=['metrics']),
        ... ])
        >>>
        >>> ctx = Context()
        >>> pipeline.run(ctx)
        >>> print(ctx['metrics'])
    """

    def __init__(
        self,
        stages: List[Stage],
        name: str = "pipeline",
        validate: bool = True,
    ):
        """
        Initialize pipeline.

        Args:
            stages: List of stages to execute.
            name: Pipeline name for logging.
            validate: Whether to validate pipeline connectivity.

        Raises:
            ValueError: If validation fails (circular dependencies, missing inputs).
        """
        self.name = name
        self.stages = stages
        self._execution_order: Optional[List[Stage]] = None

        if validate:
            self.validate()

    def validate(self) -> None:
        """
        Validate pipeline connectivity.

        Checks:
        - No circular dependencies
        - All stage inputs are satisfied
        - No duplicate stage names
        - Pipeline is executable (has topological order)

        Raises:
            ValueError: If validation fails.
        """
        # Check duplicate names
        names = [s.name for s in self.stages]
        duplicates = [name for name in names if names.count(name) > 1]
        if duplicates:
            raise ValueError(f"Duplicate stage names: {set(duplicates)}")

        # Build dependency graph
        try:
            self._compute_execution_order()
        except ValueError as e:
            raise ValueError(f"Pipeline validation failed: {e}")

        # Check that all inputs can be satisfied
        available_outputs = set()
        for stage in self.get_execution_order():
            # Check inputs
            for inp in stage.inputs:
                if inp not in available_outputs:
                    logger.warning(
                        f"Stage '{stage.name}' requires input '{inp}' "
                        f"which is not produced by any previous stage. "
                        f"This input must be provided in the initial context."
                    )
            # Add outputs
            available_outputs.update(stage.outputs)

    def _compute_execution_order(self) -> List[Stage]:
        """
        Compute topological order of stages using Kahn's algorithm.

        Returns:
            List of stages in execution order.

        Raises:
            ValueError: If circular dependencies exist.
        """
        # Build dependency graph
        # A stage depends on another if it requires its outputs
        stage_by_name = {s.name: s for s in self.stages}
        outputs_to_stage = defaultdict(list)

        # Map: output key -> stages that produce it
        for stage in self.stages:
            for output in stage.outputs:
                outputs_to_stage[output].append(stage)

        # Build adjacency list: stage -> stages that depend on it
        graph = defaultdict(list)
        in_degree = {s.name: 0 for s in self.stages}

        for stage in self.stages:
            for inp in stage.inputs:
                # Find stages that produce this input
                producers = outputs_to_stage.get(inp, [])
                for producer in producers:
                    if producer.name != stage.name:  # No self-loops
                        graph[producer.name].append(stage.name)
                        in_degree[stage.name] += 1

        # Kahn's algorithm for topological sort
        queue = deque([name for name, degree in in_degree.items() if degree == 0])
        order = []

        while queue:
            current = queue.popleft()
            order.append(stage_by_name[current])

            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check for cycles
        if len(order) != len(self.stages):
            remaining = [name for name, degree in in_degree.items() if degree > 0]
            raise ValueError(f"Circular dependencies detected involving stages: {remaining}")

        return order

    def get_execution_order(self) -> List[Stage]:
        """
        Get execution order of stages.

        Returns:
            List of stages in topological order.
        """
        if self._execution_order is None:
            self._execution_order = self._compute_execution_order()
        return self._execution_order

    def run(
        self,
        context: Optional[Context] = None,
        skip_stages: Optional[Set[str]] = None,
    ) -> Context:
        """
        Execute the pipeline.

        Args:
            context: Shared context (created if None).
            skip_stages: Set of stage names to skip.

        Returns:
            The context after execution.

        Raises:
            KeyError: If a stage's required inputs are missing.
            Exception: Any exception raised by stages.
        """
        if context is None:
            context = Context()

        skip_stages = skip_stages or set()
        execution_order = self.get_execution_order()

        logger.info(f"Starting pipeline '{self.name}' with {len(execution_order)} stages")

        for i, stage in enumerate(execution_order, 1):
            if not stage.enabled:
                logger.info(f"[{i}/{len(execution_order)}] Skipping disabled stage '{stage.name}'")
                continue

            if stage.name in skip_stages:
                logger.info(f"[{i}/{len(execution_order)}] Skipping stage '{stage.name}' (user requested)")
                continue

            logger.info(f"[{i}/{len(execution_order)}] Executing stage '{stage.name}'")

            try:
                # Setup
                if not stage._is_setup:
                    stage.setup(context)

                # Validate inputs
                stage.validate_inputs(context)

                # Execute
                stage.execute(context)

                # Cleanup
                if not stage._is_cleaned_up:
                    stage.cleanup(context)

                logger.info(f"[{i}/{len(execution_order)}] Stage '{stage.name}' completed successfully")

            except Exception as e:
                logger.error(f"[{i}/{len(execution_order)}] Stage '{stage.name}' failed: {e}")
                raise

        logger.info(f"Pipeline '{self.name}' completed successfully")
        return context

    def visualize(self) -> str:
        """
        Create a text visualization of the pipeline DAG.

        Returns:
            String representation of the pipeline graph.
        """
        lines = [f"Pipeline: {self.name}", "=" * 50]

        execution_order = self.get_execution_order()

        for i, stage in enumerate(execution_order, 1):
            status = "✓" if stage.enabled else "✗"
            lines.append(f"\n{i}. [{status}] {stage.name}")

            if stage.inputs:
                lines.append(f"   Inputs:  {', '.join(stage.inputs)}")
            if stage.outputs:
                lines.append(f"   Outputs: {', '.join(stage.outputs)}")

        return "\n".join(lines)

    def get_stage(self, name: str) -> Optional[Stage]:
        """
        Get stage by name.

        Args:
            name: Stage name.

        Returns:
            Stage instance or None if not found.
        """
        for stage in self.stages:
            if stage.name == name:
                return stage
        return None

    def disable_stage(self, name: str) -> None:
        """
        Disable a stage.

        Args:
            name: Stage name to disable.
        """
        stage = self.get_stage(name)
        if stage:
            stage.config.enabled = False

    def enable_stage(self, name: str) -> None:
        """
        Enable a stage.

        Args:
            name: Stage name to enable.
        """
        stage = self.get_stage(name)
        if stage:
            stage.config.enabled = True

    def __repr__(self) -> str:
        """String representation."""
        return f"Pipeline(name='{self.name}', stages={len(self.stages)})"
