from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from pioneerml.common.data_loader.config import DataFlowConfig
from pioneerml.common.data_writer import BaseDataWriter


@dataclass(frozen=True)
class InferenceSourcePayload:
    source_idx: int
    src_path: Path
    num_rows: int
    source_event_offset: int


@dataclass(frozen=True)
class InferenceRuntimePayload:
    cfg: dict
    device: torch.device
    model_path: str
    scripted: object
    writer: BaseDataWriter
    materialize_outputs: bool
    validated_files: list[str]
    validated_file_rows: list[int]
    flow_cfg: DataFlowConfig
    source_contexts: list[InferenceSourcePayload]
    output_path: str | None

