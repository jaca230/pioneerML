from __future__ import annotations

from pioneerml.pipeline.steps.payloads import BaseStepPayload


class HPOStepPayload(BaseStepPayload):
    REQUIRED_KEYS = ("hpo_params",)

