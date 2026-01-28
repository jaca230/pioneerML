"""
Base loader class with shared functionality.
"""

import math
import glob
import re
from pathlib import Path
from typing import Optional
from abc import ABC, abstractmethod

# Support both legacy .npy shards and new parquet shards
_CHUNK_INDEX_RE = re.compile(r"_(\d+)\.(?:npy|parquet)$")


def extract_file_index(path: Path) -> Optional[int]:
    """Return the numeric suffix from a chunk filename, if present."""
    match = _CHUNK_INDEX_RE.search(path.name)
    return int(match.group(1)) if match else None


class BaseLoader(ABC):
    """Base class for data loaders with shared file handling logic."""

    def _find_files(self, file_pattern: str, max_files: Optional[int] = None, verbose: bool = True):
        """
        Find and sort files matching the pattern.
        
        Args:
            file_pattern: Glob pattern for .npy files
            max_files: Maximum number of files to load
            verbose: Whether to print loading statistics
            
        Returns:
            List of sorted file paths
        """
        all_paths = [Path(p) for p in glob.glob(file_pattern)]
        if not all_paths:
            raise FileNotFoundError(f"No files matched pattern '{file_pattern}'")
        all_paths.sort(
            key=lambda p: (extract_file_index(p) if extract_file_index(p) is not None else math.inf, p.name)
        )
        
        if max_files is not None:
            paths = all_paths[:max_files]
            if verbose:
                import sys
                print(
                    f"Limiting to {len(paths)} files (from {len(all_paths)} total files found, max_files={max_files})",
                    file=sys.stderr,
                    flush=True,
                )
        else:
            paths = all_paths
            
        return paths

    def _extract_event_id(self, group_arr, path: Path, event_offset: int) -> int:
        """
        Extract event ID from group array or generate one.
        
        Args:
            group_arr: Group array
            path: File path
            event_offset: Event offset within the file
            
        Returns:
            Event ID
        """
        if group_arr.shape[1] > 6:
            try:
                return int(group_arr[0, 6])
            except (ValueError, TypeError):
                pass
        
        chunk_idx = extract_file_index(path) or 0
        return chunk_idx * 100000 + event_offset
    
    @staticmethod
    def _extract_file_index(path: Path) -> Optional[int]:
        """Return the numeric suffix from a chunk filename, if present."""
        return extract_file_index(path)

    @abstractmethod
    def load(self, file_pattern: str, **kwargs):
        """
        Load data from files matching the pattern.
        
        Args:
            file_pattern: Glob pattern for .npy files
            **kwargs: Loader-specific parameters
            
        Returns:
            List of record dictionaries
        """
        pass
