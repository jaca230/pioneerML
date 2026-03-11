from __future__ import annotations

from pioneerml.common.pipeline.steps.payloads import BaseStepPayload


class LoaderFactoryInitStepPayload(BaseStepPayload):
    REQUIRED_KEYS = ("loader_factory",)
