from __future__ import annotations

from pioneerml.common.pipeline.steps.payloads import BaseStepPayload


class InferenceStepPayload(BaseStepPayload):
    REQUIRED_KEYS = ("predictions_paths",)
