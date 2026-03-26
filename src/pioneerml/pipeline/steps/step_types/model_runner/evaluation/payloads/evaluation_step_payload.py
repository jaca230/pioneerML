from __future__ import annotations

from pioneerml.pipeline.steps.payloads import BaseStepPayload


class EvaluationStepPayload(BaseStepPayload):
    REQUIRED_KEYS = ("metrics",)

