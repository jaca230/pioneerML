"""
Data loaders for loading preprocessed time-group data from .npy files.

Each loader is implemented as a class for better organization and extensibility.
"""

from .base import BaseLoader
from .preprocessed_time_groups import PreprocessedTimeGroupsLoader
from .splitter_groups import SplitterGroupsLoader
from .pion_stop_groups import PionStopGroupsLoader
from .positron_angle_groups import PositronAngleGroupsLoader

# Export constants
from .constants import (
    CLASS_NAMES,
    NUM_GROUP_CLASSES,
    NODE_LABEL_TO_NAME,
    NUM_NODE_CLASSES,
    PION_MASK,
    MUON_MASK,
    POSITRON_MASK,
    ELECTRON_MASK,
    OTHER_MASK,
    BIT_TO_CLASS,
)

# Functional API for backwards compatibility
def load_preprocessed_time_groups(
    file_pattern: str,
    *,
    max_files: int | None = None,
    limit_groups: int | None = None,
    min_hits: int = 2,
    min_hits_per_label: int = 2,
    verbose: bool = True,
):
    """Load preprocessed time-group files into dictionaries for the dataset."""
    loader = PreprocessedTimeGroupsLoader()
    return loader.load(
        file_pattern,
        max_files=max_files,
        limit_groups=limit_groups,
        min_hits=min_hits,
        min_hits_per_label=min_hits_per_label,
        verbose=verbose,
    )


def load_splitter_groups(
    file_pattern: str,
    *,
    max_files: int | None = None,
    limit_groups: int | None = None,
    min_hits: int = 3,
    verbose: bool = True,
):
    """Load groups for the splitter network (per-hit classification)."""
    loader = SplitterGroupsLoader()
    return loader.load(
        file_pattern,
        max_files=max_files,
        limit_groups=limit_groups,
        min_hits=min_hits,
        verbose=verbose,
    )


def load_pion_stop_groups(
    file_pattern: str,
    *,
    pion_pdg: int = 1,
    max_files: int | None = None,
    limit_groups: int | None = None,
    min_hits: int = 3,
    min_pion_hits: int = 1,
    verbose: bool = True,
):
    """Load preprocessed pion groups for stop position regression."""
    loader = PionStopGroupsLoader()
    return loader.load(
        file_pattern,
        pion_pdg=pion_pdg,
        max_files=max_files,
        limit_groups=limit_groups,
        min_hits=min_hits,
        min_pion_hits=min_pion_hits,
        verbose=verbose,
    )


def load_positron_angle_groups(
    file_pattern: str,
    *,
    max_files: int | None = None,
    limit_groups: int | None = None,
    min_hits: int = 2,
    angle_targets_pattern: str | None = None,
    verbose: bool = True,
):
    """Load groups for positron angle regression."""
    loader = PositronAngleGroupsLoader()
    return loader.load(
        file_pattern,
        max_files=max_files,
        limit_groups=limit_groups,
        min_hits=min_hits,
        angle_targets_pattern=angle_targets_pattern,
        verbose=verbose,
    )


__all__ = [
    "BaseLoader",
    "PreprocessedTimeGroupsLoader",
    "SplitterGroupsLoader",
    "PionStopGroupsLoader",
    "PositronAngleGroupsLoader",
    "load_preprocessed_time_groups",
    "load_splitter_groups",
    "load_pion_stop_groups",
    "load_positron_angle_groups",
    "CLASS_NAMES",
    "NUM_GROUP_CLASSES",
    "NODE_LABEL_TO_NAME",
    "NUM_NODE_CLASSES",
    "PION_MASK",
    "MUON_MASK",
    "POSITRON_MASK",
    "ELECTRON_MASK",
    "OTHER_MASK",
    "BIT_TO_CLASS",
]


