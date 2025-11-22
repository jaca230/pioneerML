"""
Dataset classes and data utilities for PIONEER reconstruction.

Provides:
- GraphRecord, PionStopRecord: Data structures for different tasks
- GraphGroupDataset: Standard time-group classification/regression
- PionStopGraphDataset: Specialized for pion stop position regression
- SplitterGraphDataset: Per-hit classification for multi-particle groups
- Graph construction utilities (fully_connected_edge_index, build_edge_attr)
"""

from pioneerml.data.datasets import (
    GraphRecord,
    PionStopRecord,
    GraphGroupDataset,
    PionStopGraphDataset,
    SplitterGraphDataset,
    fully_connected_edge_index,
    build_edge_attr,
)

__all__ = [
    "GraphRecord",
    "PionStopRecord",
    "GraphGroupDataset",
    "PionStopGraphDataset",
    "SplitterGraphDataset",
    "fully_connected_edge_index",
    "build_edge_attr",
]
