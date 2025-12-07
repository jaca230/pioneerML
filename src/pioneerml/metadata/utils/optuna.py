"""
Optuna study serialization utilities.
"""

from __future__ import annotations

from typing import Any


def serialize_optuna_study(
    study: Any,
    *,
    include_failed: bool = False,
    include_intermediate_values: bool = False,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """
    Convert an Optuna study into a lightweight list of trial dicts suitable
    for storing in TrainingMetadata.hyperparameter_history.
    
    Args:
        study: Optuna Study object
        include_failed: Whether to include failed trials
        include_intermediate_values: Whether to include intermediate values
        limit: Limit number of trials to return (None = all)
    
    Returns:
        List of trial dictionaries
    """
    try:
        import optuna  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Optuna is required to serialize study history.") from e

    trials = study.get_trials(deepcopy=False)
    # sort by trial number for reproducibility
    trials = sorted(trials, key=lambda t: t.number)
    if not include_failed:
        trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    if limit is not None:
        trials = trials[-limit:]

    history: list[dict[str, Any]] = []
    for t in trials:
        entry = {
            "number": t.number,
            "state": t.state.name,
            "value": t.value,
            "params": dict(t.params),
            "user_attrs": dict(t.user_attrs),
            "datetime_start": t.datetime_start.isoformat() if t.datetime_start else None,
            "datetime_complete": t.datetime_complete.isoformat() if t.datetime_complete else None,
        }
        if include_intermediate_values:
            entry["intermediate_values"] = dict(t.intermediate_values)
        history.append(entry)
    return history


