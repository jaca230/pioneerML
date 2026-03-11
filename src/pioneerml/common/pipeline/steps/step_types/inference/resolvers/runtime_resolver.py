from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from pioneerml.common.data_loader.config import DataFlowConfig
from pioneerml.common.data_loader.input_source import InputSourceSet
from pioneerml.common.pipeline.payloads import InferenceSourcePayload

from ....resolver import BaseConfigResolver


class InferenceRuntimeResolver(BaseConfigResolver):
    def resolve(self, *, cfg: dict[str, Any]) -> None:
        cfg["inference_config"] = dict(cfg)

    @staticmethod
    def resolve_device(*, prefer_cuda: bool = True) -> torch.device:
        if prefer_cuda and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def load_torchscript(*, model_path: str, device: torch.device):
        scripted = torch.jit.load(model_path, map_location=device)
        scripted.eval()
        return scripted

    @staticmethod
    def resolve_materialize_outputs(
        cfg: dict,
        *,
        inputs: dict,
        default_for_train_mode: bool = True,
    ) -> bool:
        explicit = cfg.get("materialize_outputs", None)
        if explicit is not None:
            return bool(explicit)
        if not default_for_train_mode:
            return False
        mode = str(inputs.get("mode", "inference")).strip().lower()
        return mode == "train"

    @staticmethod
    def resolve_validated_files(inputs: dict) -> list[str]:
        validated_files = [
            str(path)
            for path in (
                inputs.get("validated_files")
                or inputs.get("main_sources")
                or inputs.get("parquet_paths")
                or []
            )
        ]
        if not validated_files:
            raise RuntimeError("No validated files provided for inference.")
        return validated_files

    @staticmethod
    def resolve_validated_file_rows(inputs: dict, *, total_files: int) -> list[int]:
        rows = [int(v) for v in (inputs.get("validated_file_rows") or [])]
        if len(rows) != int(total_files):
            raise RuntimeError(
                f"Expected validated_file_rows aligned with validated_files ({total_files}), got {len(rows)}."
            )
        return rows

    @staticmethod
    def resolve_data_flow_config(
        inputs: dict,
        *,
        default_batch_size: int = 64,
        default_row_groups_per_chunk: int = 4,
        default_num_workers: int = 0,
    ) -> DataFlowConfig:
        flow_cfg_dict = dict(inputs.get("data_flow_config") or {})
        return DataFlowConfig(
            batch_size=int(flow_cfg_dict.get("batch_size", inputs.get("batch_size", default_batch_size))),
            row_groups_per_chunk=int(
                flow_cfg_dict.get("row_groups_per_chunk", inputs.get("row_groups_per_chunk", default_row_groups_per_chunk))
            ),
            num_workers=int(flow_cfg_dict.get("num_workers", inputs.get("num_workers", default_num_workers))),
        )

    @staticmethod
    def iter_source_contexts(*, validated_files: list[str], validated_file_rows: list[int]) -> list[InferenceSourcePayload]:
        out: list[InferenceSourcePayload] = []
        if len(validated_file_rows) != len(validated_files):
            raise RuntimeError(
                "validated_file_rows must align with validated_files. "
                f"Got {len(validated_file_rows)} vs {len(validated_files)}."
            )
        source_event_offset = 0
        for source_idx, (src_file, num_rows) in enumerate(zip(validated_files, validated_file_rows, strict=True)):
            src_path = Path(src_file).expanduser().resolve()
            out.append(
                InferenceSourcePayload(
                    source_idx=int(source_idx),
                    src_path=src_path,
                    num_rows=num_rows,
                    source_event_offset=int(source_event_offset),
                )
            )
            source_event_offset += num_rows
        return out

    @staticmethod
    def select_source_aligned_paths(
        paths: list[str] | None,
        *,
        source_index: int,
        total_files: int,
        label: str,
    ) -> list[str] | None:
        if not paths:
            return None
        if len(paths) != int(total_files):
            raise RuntimeError(
                f"Expected {total_files} {label} files aligned with primary inputs, got {len(paths)}."
            )
        return [str(paths[source_index])]

    @classmethod
    def resolve_optional_source_paths(
        cls,
        *,
        inputs: dict,
        source_index: int,
        total_files: int,
        source_name: str,
        input_keys: tuple[str, ...],
    ) -> list[str] | None:
        source_paths: list[str] | None = None
        for key in input_keys:
            raw = inputs.get(key)
            if raw:
                source_paths = [str(path) for path in raw]
                break
        return cls.select_source_aligned_paths(
            source_paths,
            source_index=source_index,
            total_files=total_files,
            label=source_name,
        )

    @staticmethod
    def build_input_source_for_path(
        *,
        src_path: Path,
        optional_sources: dict[str, list[str] | None] | None = None,
    ) -> InputSourceSet:
        dynamic_sources = {
            str(k): (list(v) if v is not None else None)
            for k, v in dict(optional_sources or {}).items()
        }
        return InputSourceSet(
            main_sources=[str(src_path)],
            optional_sources_by_name=dynamic_sources,
        )
