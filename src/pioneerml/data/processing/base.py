"""
Abstract base class for data processors.

Processors operate on already loaded tabular data (e.g., Polars DataFrames),
adding derived features and/or converting to model-ready records.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import polars as pl


class BaseProcessor(ABC):
    """Minimal contract for dataset processors."""

    @abstractmethod
    def process(self, df: pl.DataFrame) -> Any:
        """Transform an in-memory DataFrame into model-ready objects."""
