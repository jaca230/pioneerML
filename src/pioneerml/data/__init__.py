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
    PositronAngleDataset,
    fully_connected_edge_index,
    build_edge_attr,
)
from pioneerml.data.loaders import (
    load_preprocessed_time_groups,
    load_splitter_groups,
    load_pion_stop_groups,
    load_positron_angle_groups,
    CLASS_NAMES,
    NUM_GROUP_CLASSES,
    NODE_LABEL_TO_NAME,
    NUM_NODE_CLASSES,
)

__all__ = [
    "GraphRecord",
    "PionStopRecord",
    "GraphGroupDataset",
    "PionStopGraphDataset",
    "SplitterGraphDataset",
    "PositronAngleDataset",
    "fully_connected_edge_index",
    "build_edge_attr",
    "load_preprocessed_time_groups",
    "load_splitter_groups",
    "load_pion_stop_groups",
    "load_positron_angle_groups",
    "CLASS_NAMES",
    "NUM_GROUP_CLASSES",
    "NODE_LABEL_TO_NAME",
    "NUM_NODE_CLASSES",
]
