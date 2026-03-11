from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WriterSetupPayload:
    writer_name: str
    output_backend_name: str
    output_dir: str | None
    output_path: str | None
    fallback_output_dir: str
    streaming: bool
    write_timestamped: bool
    timestamp: str | None = None

