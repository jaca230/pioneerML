from __future__ import annotations

from pioneerml.pipeline.steps.payloads import BaseStepPayload


class ExportStepPayload(BaseStepPayload):
    REQUIRED_KEYS = ("torchscript_path", "metadata_path")

