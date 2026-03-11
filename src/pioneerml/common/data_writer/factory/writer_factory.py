from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pioneerml.common.data_writer.base_data_writer import BaseDataWriter
from pioneerml.common.data_writer.config import WriterRunConfig

from .registry import resolve_writer


@dataclass(frozen=True)
class WriterFactory:
    writer_cls: type[BaseDataWriter] | None = None
    writer_name: str | None = None
    output_backend_name: str = "parquet"
    run_config: WriterRunConfig | None = None
    writer_params: dict[str, Any] | None = None

    def _resolve_writer_class(self) -> type[BaseDataWriter]:
        if self.writer_cls is not None:
            return self.writer_cls
        if self.writer_name is None:
            raise RuntimeError("WriterFactory requires either writer_cls or writer_name.")
        return resolve_writer(self.writer_name)

    def create(self) -> BaseDataWriter:
        writer_cls = self._resolve_writer_class()
        if self.run_config is None:
            raise RuntimeError("WriterFactory requires run_config.")
        if not hasattr(writer_cls, "from_factory"):
            raise RuntimeError(
                f"{writer_cls.__name__} must implement classmethod from_factory(...) for WriterFactory."
            )
        return writer_cls.from_factory(
            output_backend_name=str(self.output_backend_name),
            run_config=self.run_config,
            writer_params=dict(self.writer_params or {}),
        )
