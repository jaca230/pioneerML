"""Parquet helpers for common data access patterns."""

from .chunked_reader import ParquetChunkReader

__all__ = ["ParquetChunkReader"]
