from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WriterRunConfig:
    output_dir: Path
    timestamp: str
    streaming: bool
    write_timestamped: bool

