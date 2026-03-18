from __future__ import annotations

from .payloads import ModelHandleBuilderStepPayload
from .resolvers import ModelHandleBuilderConfigResolver, ModelHandleBuilderRuntimeStateResolver

from ..base_pipeline_step import BasePipelineStep


class BaseModelHandleBuilderStep(BasePipelineStep):
    MODEL_REPO_PARENTS_UP = ModelHandleBuilderRuntimeStateResolver.MODEL_REPO_PARENTS_UP_DEFAULT
    DEFAULT_CONFIG = {"model_path": None, "model_type": "script", "model_subdir": None}
    config_resolver_classes = (ModelHandleBuilderConfigResolver,)
    payload_resolver_classes = (ModelHandleBuilderRuntimeStateResolver,)

    def _build_payload(
        self,
        *,
        model_handle: object,
    ) -> ModelHandleBuilderStepPayload:
        return ModelHandleBuilderStepPayload(model_handle=model_handle)

    def _execute(self) -> ModelHandleBuilderStepPayload:
        model_handle = self.runtime_state.get("model_handle")
        if model_handle is None:
            raise RuntimeError(f"{self.__class__.__name__} runtime_state missing valid 'model_handle'.")
        return self._build_payload(model_handle=model_handle)
