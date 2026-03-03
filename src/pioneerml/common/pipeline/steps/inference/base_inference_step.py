from __future__ import annotations

from dataclasses import dataclass
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
import torch

from pioneerml.common.loader.config import DataFlowConfig
from pioneerml.common.parquet import ParquetInputSet

from ..base_pipeline_step import BasePipelineStep


@dataclass(frozen=True)
class InferenceFileContext:
    file_idx: int
    src_path: Path
    num_rows: int
    file_event_offset: int


class BaseInferenceStep(BasePipelineStep):
    @staticmethod
    def resolve_device(prefer_cuda: bool = True) -> torch.device:
        if prefer_cuda and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def load_torchscript(model_path: str, *, device: torch.device):
        scripted = torch.jit.load(model_path, map_location=device)
        scripted.eval()
        return scripted

    @staticmethod
    def resolve_streaming_flag(cfg: dict, *, default: bool = True) -> bool:
        return bool(cfg.get("streaming", default))

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
    def create_streaming_tmp_dir(tmp_root: str | Path, *, prefix: str = "run_") -> Path:
        root = Path(tmp_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=prefix, dir=str(root)))

    @staticmethod
    def write_streaming_table(*, table, dst_path: str | Path) -> None:
        out = Path(dst_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out)

    @staticmethod
    def resolve_validated_files(inputs: dict) -> list[str]:
        validated_files = [str(p) for p in inputs.get("validated_files") or inputs.get("parquet_paths") or []]
        if not validated_files:
            raise RuntimeError("No validated files provided for inference.")
        return validated_files

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
    def iter_file_contexts(validated_files: list[str]) -> list[InferenceFileContext]:
        out: list[InferenceFileContext] = []
        file_event_offset = 0
        for file_idx, src_file in enumerate(validated_files):
            src_path = Path(src_file).expanduser().resolve()
            num_rows = int(pq.ParquetFile(str(src_path)).metadata.num_rows)
            out.append(
                InferenceFileContext(
                    file_idx=int(file_idx),
                    src_path=src_path,
                    num_rows=num_rows,
                    file_event_offset=int(file_event_offset),
                )
            )
            file_event_offset += num_rows
        return out

    @staticmethod
    def select_file_aligned_paths(
        paths: list[str] | None,
        *,
        file_index: int,
        total_files: int,
        label: str,
    ) -> list[str] | None:
        if not paths:
            return None
        if len(paths) != int(total_files):
            raise RuntimeError(
                f"Expected {total_files} {label} files aligned with primary inputs, got {len(paths)}."
            )
        return [str(paths[file_index])]

    def resolve_optional_file_paths(
        self,
        *,
        inputs: dict,
        file_index: int,
        total_files: int,
        source_name: str,
        input_keys: tuple[str, ...],
    ) -> list[str] | None:
        source_paths: list[str] | None = None
        for key in input_keys:
            raw = inputs.get(key)
            if raw:
                source_paths = [str(p) for p in raw]
                break
        return self.select_file_aligned_paths(
            source_paths,
            file_index=file_index,
            total_files=total_files,
            label=source_name,
        )

    def build_parquet_input_for_file(
        self,
        *,
        src_path: Path,
        optional_sources: dict[str, list[str] | None] | None = None,
    ) -> ParquetInputSet:
        dynamic_sources = {
            str(k): (list(v) if v is not None else None)
            for k, v in dict(optional_sources or {}).items()
        }
        return ParquetInputSet(
            main_paths=[str(src_path)],
            optional_paths_by_name=dynamic_sources,
        )

    @staticmethod
    def append_streamed_file_record(
        *,
        records: list[dict],
        src_path: Path,
        prediction_path: Path,
        num_rows: int,
    ) -> None:
        records.append(
            {
                "source_path": str(src_path),
                "prediction_path": str(prediction_path),
                "num_rows": int(num_rows),
            }
        )
