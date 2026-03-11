from __future__ import annotations

from dataclasses import dataclass

from pioneerml.common.data_loader.config import DataFlowConfig


@dataclass(frozen=True)
class LoaderRuntimePayload:
    mode: str
    data_flow_config: DataFlowConfig

