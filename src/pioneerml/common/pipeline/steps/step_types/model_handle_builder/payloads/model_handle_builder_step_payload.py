from __future__ import annotations

from pioneerml.common.pipeline.steps.payloads import BaseStepPayload


class ModelHandleBuilderStepPayload(BaseStepPayload):
    REQUIRED_KEYS = ("model_handle",)
