"""
Data loaders for loading preprocessed time-group data from .npy files.

Each loader is implemented as a class for better organization and extensibility.
"""

from .base import BaseLoader
from .hits_info import HitsAndInfoLoader

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
def load_hits_and_info(
    hits_pattern: str,
    info_pattern: str,
    *,
    max_files: int | None = None,
    limit_groups: int | None = None,
    min_hits: int = 2,
    verbose: bool = True,
    include_hit_labels: bool = False,
):
    """Load paired hits/info NPY dumps into GraphRecords."""
    loader = HitsAndInfoLoader()
    return loader.load(
        hits_pattern,
        info_pattern,
        max_files=max_files,
        limit_groups=limit_groups,
        min_hits=min_hits,
        verbose=verbose,
        include_hit_labels=include_hit_labels,
    )


__all__ = [
    "BaseLoader",
    "HitsAndInfoLoader",
    "load_hits_and_info",
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
