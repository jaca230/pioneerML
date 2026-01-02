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
    build_event_graph,
)
from pioneerml.data.loaders import (
    load_hits_and_info,
    CLASS_NAMES,
    NUM_GROUP_CLASSES,
    NODE_LABEL_TO_NAME,
    NUM_NODE_CLASSES,
)
from pioneerml.data.event_mixer import EventMixer, MixedEventDataset, EventContainer, save_mixed_events

__all__ = [
    "GraphRecord",
    "PionStopRecord",
    "GraphGroupDataset",
    "PionStopGraphDataset",
    "SplitterGraphDataset",
    "PositronAngleDataset",
    "fully_connected_edge_index",
    "build_edge_attr",
    "build_event_graph",
    "load_hits_and_info",
    "CLASS_NAMES",
    "NUM_GROUP_CLASSES",
    "NODE_LABEL_TO_NAME",
    "NUM_NODE_CLASSES",
    "EventMixer",
    "MixedEventDataset",
    "EventContainer",
    "save_mixed_events",
]
