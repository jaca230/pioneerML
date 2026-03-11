from __future__ import annotations

from pioneerml.common.pipeline.steps.payloads import BaseStepPayload


class TrainingStepPayload(BaseStepPayload):
    REQUIRED_KEYS = ("module", "training_context")
