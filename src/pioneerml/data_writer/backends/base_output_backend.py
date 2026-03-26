from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pyarrow as pa


class OutputBackend(ABC):
    """Backend contract for writing output tables."""

    @abstractmethod
    def default_extension(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def write_table_atomic(self, *, table: pa.Table, dst_path: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def open_sink(self, *, dst_path: Path) -> Any:
        """Open a streaming sink handle for chunk appends."""
        raise NotImplementedError

    @abstractmethod
    def append_chunk(self, *, sink: Any, table: pa.Table) -> None:
        """Append one chunk table to a previously opened sink."""
        raise NotImplementedError

    @abstractmethod
    def close_sink(self, *, sink: Any) -> None:
        """Close and finalize a previously opened sink."""
        raise NotImplementedError
