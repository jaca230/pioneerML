from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
import json
from typing import Any, TypeAlias

from ..payloads import BaseStepPayload, StepPayloads
from ..resolver import BaseConfigResolver, BasePayloadResolver, DefaultConfigResolver

JSONPrimitive: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONPrimitive | list["JSONValue"] | dict[str, "JSONValue"]
JSONObject: TypeAlias = dict[str, JSONValue]


class BasePipelineStep(ABC):
    step_key: str | None = None
    DEFAULT_CONFIG: JSONObject = {}
    # Preferred split: config resolvers run at init, payload resolvers run at execute.
    config_resolver_classes: tuple[type[BaseConfigResolver], ...] = ()
    payload_resolver_classes: tuple[type[BasePayloadResolver], ...] = ()
    # Legacy alias (treated as config resolvers when split is not provided).
    resolver_classes: tuple[type[BaseConfigResolver], ...] = ()

    def __init__(self, *, pipeline_config: JSONObject | None = None) -> None:
        self.pipeline_config = pipeline_config
        self.runtime_state: dict[str, Any] = {}
        self.config_json: JSONObject = self.resolve_config()

    def default_config(self) -> JSONObject:
        return {}

    def resolve_config(self) -> JSONObject:
        if not isinstance(self.step_key, str) or not self.step_key:
            raise RuntimeError(f"{self.__class__.__name__} must define non-empty step_key.")
        if self.pipeline_config is None:
            cfg: JSONObject = {}
        else:
            if not isinstance(self.pipeline_config, dict):
                raise TypeError(
                    f"Expected mapping config, got {type(self.pipeline_config).__name__}."
                )
            raw = self.pipeline_config.get(self.step_key)
            if raw is None:
                cfg = {}
            elif isinstance(raw, dict):
                cfg = dict(raw)
            else:
                raise TypeError(
                    f"Expected dict for '{self.step_key}' config, got {type(raw).__name__}."
                )
        for resolver_cls in (DefaultConfigResolver,) + self._config_resolver_chain():
            resolver = resolver_cls(step=self)
            resolver.resolve(cfg=cfg)
        try:
            json.dumps(cfg)
        except TypeError as exc:
            raise TypeError(
                f"{self.__class__.__name__} produced non-JSON config. "
                "Resolvers must write runtime objects to `runtime_state`."
            ) from exc
        return cfg

    def _config_resolver_chain(self) -> tuple[type[BaseConfigResolver], ...]:
        if self.config_resolver_classes:
            return tuple(self.config_resolver_classes)
        return tuple(self.resolver_classes)

    def _payload_resolver_chain(self) -> tuple[type[BasePayloadResolver], ...]:
        return tuple(self.payload_resolver_classes)

    def resolve_payload(self, *, payloads: Mapping[str, Any] | None = None) -> None:
        payload_container = StepPayloads.from_mapping(payloads)
        for resolver_cls in self._payload_resolver_chain():
            resolver = resolver_cls(step=self)
            resolver.resolve(payloads=payload_container, runtime_state=self.runtime_state)

    @abstractmethod
    def _execute(self, *args, **kwargs):
        raise NotImplementedError

    def execute(self, *args, payloads: Mapping[str, Any] | None = None, **kwargs) -> BaseStepPayload:
        self.resolve_payload(payloads=payloads)
        out = self._execute(*args, **kwargs)
        if not isinstance(out, BaseStepPayload):
            raise TypeError(
                f"{self.__class__.__name__}.execute() must return BaseStepPayload (or subclass), "
                f"got {type(out).__name__}."
            )
        return out
