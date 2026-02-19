from __future__ import annotations

import tempfile
from pathlib import Path

import pyarrow.parquet as pq
import torch

from ..base_pipeline_service import BasePipelineService


class BaseInferenceService(BasePipelineService):
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
