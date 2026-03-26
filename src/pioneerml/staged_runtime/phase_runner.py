from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from .stage_runner import StageRunner


class PhaseRunner:
    """Orchestrates named phase runners on top of linear StageRunner instances."""

    def __init__(self) -> None:
        self._phase_runners: dict[str, StageRunner] = {}

    def register_phase(self, *, name: str, runner: StageRunner) -> None:
        key = str(name).strip()
        if not key:
            raise ValueError("Phase name must be non-empty.")
        self._phase_runners[key] = runner

    def run_phase(
        self,
        *,
        name: str,
        state: MutableMapping[str, Any],
        owner: Any,
        context_fields: dict[str, Any] | None = None,
    ) -> MutableMapping[str, Any]:
        key = str(name).strip()
        if key not in self._phase_runners:
            raise KeyError(f"Unknown phase '{name}'. Registered phases: {sorted(self._phase_runners.keys())}")
        runner = self._phase_runners[key]
        runtime_context_fields = dict(context_fields or {})
        out = runner.run(
            state=state,
            owner=owner,
            context_fields=runtime_context_fields,
        )
        run_index = int(runtime_context_fields.get("run_index", runtime_context_fields.get("chunk_index", 0)))
        runner.observer.on_run_end(
            run_index=run_index,
            phase=key,
            state=out,
            owner=owner,
        )
        return out

