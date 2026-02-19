from __future__ import annotations

from abc import ABC, abstractmethod


class BasePipelineService(ABC):
    step_key: str | None = None

    def __init__(self, *, pipeline_config: dict | None = None) -> None:
        self.pipeline_config = pipeline_config

    @abstractmethod
    def default_config(self) -> dict:
        raise NotImplementedError

    def validate_config(self, cfg: dict) -> dict:
        return cfg

    @staticmethod
    def merge_config(base: dict, override: dict | None) -> dict:
        merged = dict(base)
        if override is not None:
            merged.update({k: v for k, v in dict(override).items() if v is not None})
        return merged

    def step_config(self) -> dict | None:
        if self.pipeline_config is None:
            return None
        if not isinstance(self.pipeline_config, dict):
            raise TypeError(f"Expected mapping config, got {type(self.pipeline_config).__name__}.")
        if not isinstance(self.step_key, str) or not self.step_key:
            raise RuntimeError(f"{self.__class__.__name__} must define non-empty step_key.")
        raw = self.pipeline_config.get(self.step_key)
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise TypeError(f"Expected dict for '{self.step_key}' config, got {type(raw).__name__}.")
        return dict(raw)

    def get_config(self) -> dict:
        cfg = self.merge_config(self.default_config(), self.step_config())
        return self.validate_config(cfg)

