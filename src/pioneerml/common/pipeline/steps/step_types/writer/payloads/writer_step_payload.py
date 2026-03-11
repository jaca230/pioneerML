from __future__ import annotations

from pioneerml.common.pipeline.steps.payloads import BaseStepPayload


class WriterStepPayload(BaseStepPayload):
    REQUIRED_KEYS = ("writer_setup",)

