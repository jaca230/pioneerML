from __future__ import annotations

from abc import abstractmethod

from ..base_pipeline_service import BasePipelineService


class BaseEvaluationService(BasePipelineService):
    @abstractmethod
    def execute(self) -> dict:
        raise NotImplementedError

