from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Iterator
from collections.abc import Callable
from typing import Any

from .backends import OutputBackend, create_output_backend
from .input_source import PredictionSet


class BaseDataWriter:
    @classmethod
    def from_factory(
        cls,
        *,
        output_backend_name: str,
        run_config: Any | None = None,
        writer_params: dict[str, Any] | None = None,
    ):
        _ = run_config
        params = dict(writer_params or {})
        output_backend = params.get("output_backend")
        if output_backend is None:
            output_backend = create_output_backend(output_backend_name, config={})
        return cls(output_backend=output_backend)

    def __init__(
        self,
        *,
        output_backend: OutputBackend | None = None,
        output_backend_name: str = "parquet",
    ) -> None:
        self.output_backend = output_backend if output_backend is not None else create_output_backend(output_backend_name)

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

    def write_table(self, *, table, dst_path: Path) -> None:
        self.output_backend.write_table_atomic(table=table, dst_path=dst_path)

    def resolve_prediction_output_path(
        self,
        *,
        src_path: Path | None,
        output_dir: Path,
        output_path: str | None,
    ) -> Path:
        if output_path:
            return Path(output_path).expanduser().resolve()
        stem = src_path.stem if src_path is not None else "preds"
        return output_dir / f"{stem}_preds{self.output_backend.default_extension()}"

    def write_table_with_optional_timestamp(
        self,
        *,
        table,
        pred_path: Path,
        output_dir: Path,
        src_path: Path | None,
        write_timestamped: bool,
        timestamp: str,
    ) -> tuple[str, str | None]:
        self.write_table(table=table, dst_path=pred_path)
        ts_path: str | None = None
        if write_timestamped:
            stem = src_path.stem if src_path is not None else "preds"
            timestamped = output_dir / f"{stem}_preds_{timestamp}{self.output_backend.default_extension()}"
            self.write_timestamped_copy(pred_path, timestamped)
            ts_path = str(timestamped)
        return str(pred_path), ts_path

    def build_prediction_set(
        self,
        *,
        batch,
        model_output: Any,
        src_path: Path,
        num_rows: int,
        cfg: Mapping[str, Any] | None = None,
    ) -> PredictionSet:
        _ = batch
        _ = model_output
        _ = src_path
        _ = num_rows
        _ = cfg
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement build_prediction_set(...) for inference usage."
        )

    def write_partitioned_tables(
        self,
        *,
        validated_files: list[str],
        validated_file_rows: list[int] | None,
        output_dir: Path,
        output_path: str | None,
        write_timestamped: bool,
        timestamp: str,
        build_table_for_span: Callable[[int, int, int, str], object],
        build_table_full: Callable[[], object] | None = None,
    ) -> tuple[list[str], list[str]]:
        out_paths: list[str] = []
        ts_paths: list[str] = []

        if validated_files:
            for src_file, start, end in self.iter_validated_file_spans(
                validated_files,
                validated_file_rows=validated_file_rows,
            ):
                n_rows = int(end - start)
                table = build_table_for_span(start, end, n_rows, src_file)
                pred_path = (
                    Path(output_path).expanduser().resolve()
                    if (output_path and len(validated_files) == 1)
                    else output_dir / f"{Path(src_file).stem}_preds{self.output_backend.default_extension()}"
                )
                pred, ts = self.write_table_with_optional_timestamp(
                    table=table,
                    pred_path=pred_path,
                    output_dir=output_dir,
                    src_path=Path(src_file),
                    write_timestamped=write_timestamped,
                    timestamp=timestamp,
                )
                out_paths.append(pred)
                if ts is not None:
                    ts_paths.append(ts)
            return out_paths, ts_paths

        if build_table_full is None:
            raise ValueError("build_table_full is required when validated_files is empty.")
        table = build_table_full()
        pred_path = self.resolve_prediction_output_path(
            src_path=None,
            output_dir=output_dir,
            output_path=output_path,
        )
        pred, ts = self.write_table_with_optional_timestamp(
            table=table,
            pred_path=pred_path,
            output_dir=output_dir,
            src_path=None,
            write_timestamped=write_timestamped,
            timestamp=timestamp,
        )
        out_paths.append(pred)
        if ts is not None:
            ts_paths.append(ts)
        return out_paths, ts_paths

    @staticmethod
    def iter_validated_file_spans(
        validated_files: list[str],
        *,
        validated_file_rows: list[int] | None = None,
    ) -> Iterator[tuple[str, int, int]]:
        if validated_file_rows is None:
            raise ValueError("validated_file_rows are required for backend-agnostic writer spans.")
        if len(validated_file_rows) != len(validated_files):
            raise ValueError(
                "validated_file_rows must align with validated_files. "
                f"Got {len(validated_file_rows)} vs {len(validated_files)}."
            )
        start = 0
        for src_file, n_rows in zip(validated_files, validated_file_rows, strict=True):
            end = start + n_rows
            yield src_file, start, end
            start = end
