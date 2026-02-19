from __future__ import annotations

import errno
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

from ..base_pipeline_service import BasePipelineService


class BaseOutputAdapterService(BasePipelineService):
    def default_config(self) -> dict:
        return {
            "write_timestamped": False,
            "check_accuracy": False,
        }

    @staticmethod
    def timestamp() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def ensure_output_dir(output_dir: str | None, fallback: str) -> Path:
        out_dir = Path(output_dir) if output_dir else Path(fallback)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    @staticmethod
    def write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def write_timestamped_copy(src_path: Path, dst_path: Path) -> None:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)

    @staticmethod
    def atomic_promote_file(*, src_path: Path, dst_path: Path) -> None:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        part_path = dst_path.with_suffix(dst_path.suffix + ".part")
        if part_path.exists():
            part_path.unlink()
        try:
            os.replace(src_path, part_path)
            os.replace(part_path, dst_path)
        except OSError as exc:
            if exc.errno == errno.EXDEV:
                raise RuntimeError(
                    "Atomic promotion failed across filesystems (EXDEV). "
                    "Configure temporary and output paths on the same filesystem."
                ) from exc
            raise

    @staticmethod
    def atomic_write_table(*, table, dst_path: Path) -> None:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        part_path = dst_path.with_suffix(dst_path.suffix + ".part")
        if part_path.exists():
            part_path.unlink()
        pq.write_table(table, part_path)
        os.replace(part_path, dst_path)

    @staticmethod
    def cleanup_directory(path: str | Path | None) -> None:
        if not path:
            return
        shutil.rmtree(str(path), ignore_errors=True)

    def promote_streamed_prediction_files(
        self,
        *,
        streamed_entries: list[dict],
        validated_files: list[str],
        output_dir: Path,
        output_path: str | None,
        write_timestamped: bool,
        timestamp: str,
        cleanup_streaming_tmp: bool = True,
        streaming_tmp_dir: str | Path | None = None,
    ) -> tuple[list[str], list[str]]:
        if output_path and len(streamed_entries) != 1:
            raise ValueError("output_path is only supported when exactly one input parquet file is provided.")

        per_file_output_paths: list[str] = []
        per_file_timestamped_paths: list[str] = []
        for idx, entry in enumerate(streamed_entries):
            src_file = str(entry.get("source_path") or (validated_files[idx] if idx < len(validated_files) else "unknown.parquet"))
            streamed_pred_path = Path(str(entry["prediction_path"])).expanduser().resolve()
            pred_path = (
                Path(output_path).expanduser().resolve()
                if (output_path and len(streamed_entries) == 1)
                else output_dir / f"{Path(src_file).stem}_preds.parquet"
            )
            self.atomic_promote_file(src_path=streamed_pred_path, dst_path=pred_path)
            per_file_output_paths.append(str(pred_path))

            if write_timestamped:
                timestamped = output_dir / f"{Path(src_file).stem}_preds_{timestamp}.parquet"
                self.write_timestamped_copy(pred_path, timestamped)
                per_file_timestamped_paths.append(str(timestamped))

        if cleanup_streaming_tmp:
            self.cleanup_directory(streaming_tmp_dir)
        return per_file_output_paths, per_file_timestamped_paths
