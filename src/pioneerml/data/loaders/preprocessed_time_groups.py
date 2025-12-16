"""
Loader for preprocessed time-group classification data.
"""

from typing import List, Dict, Any, Optional
import numpy as np

from .base import BaseLoader
from .constants import BIT_TO_CLASS, NUM_GROUP_CLASSES, CLASS_NAMES


def _labels_from_mask(mask: int) -> List[int]:
    """Decode a bitmask into class indices, collapsing e± into the MIP label."""
    labels = set()
    for bit, class_idx in BIT_TO_CLASS.items():
        if mask & bit:
            labels.add(class_idx)
    return list(labels)


def _count_labels(mask_values: np.ndarray) -> np.ndarray:
    """Count per-class hit occurrences for a group's bitmasks."""
    counts = np.zeros(NUM_GROUP_CLASSES, dtype=int)
    for mask in mask_values.astype(int):
        if mask <= 0:
            continue
        for class_idx in _labels_from_mask(mask):
            counts[class_idx] += 1
    return counts


class PreprocessedTimeGroupsLoader(BaseLoader):
    """Loader for preprocessed time-group classification data."""

    def load(
        self,
        file_pattern: str,
        *,
        max_files: Optional[int] = None,
        limit_groups: Optional[int] = None,
        min_hits: int = 2,
        min_hits_per_label: int = 2,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Load preprocessed time-group files into dictionaries for the dataset.
        
        Args:
            file_pattern: Glob pattern for .npy files (e.g., 'data/mainTimeGroups_*.npy')
            max_files: Maximum number of files to load
            limit_groups: Maximum number of groups to load across all files
            min_hits: Minimum number of hits per group
            min_hits_per_label: Minimum hits per label to include a group
            verbose: Whether to print loading statistics
            
        Returns:
            List of dictionaries with keys: coord, z, energy, view, labels, event_id
        """
        paths = self._find_files(file_pattern, max_files=max_files, verbose=verbose)

        records: List[Dict[str, Any]] = []
        label_totals = np.zeros(NUM_GROUP_CLASSES, dtype=int)

        for path in paths:
            if limit_groups is not None and len(records) >= limit_groups:
                break
            chunk = np.load(path, allow_pickle=True)
            for event_offset, event_groups in enumerate(chunk):
                if limit_groups is not None and len(records) >= limit_groups:
                    break
                if event_groups is None or len(event_groups) == 0:
                    continue
                for group_idx, group in enumerate(event_groups):
                    if limit_groups is not None and len(records) >= limit_groups:
                        break
                    group_arr = np.asarray(group)
                    if group_arr.ndim != 2 or group_arr.shape[0] < min_hits or group_arr.shape[1] < 6:
                        continue

                    mask_counts = _count_labels(group_arr[:, 5])
                    labels = [cls for cls, count in enumerate(mask_counts) if count >= min_hits_per_label]
                    if not labels:
                        labels = [cls for cls, count in enumerate(mask_counts) if count > 0]
                    if not labels:
                        continue  # ignore groups without target particles

                    record_event_id = self._extract_event_id(group_arr, path, event_offset)

                    records.append({
                        'coord': group_arr[:, 0].astype(np.float32),
                        'z': group_arr[:, 1].astype(np.float32),
                        'energy': group_arr[:, 3].astype(np.float32),
                        'view': group_arr[:, 2].astype(np.float32),
                        'labels': labels,
                        'event_id': record_event_id,
                    })
                    for lbl in labels:
                        label_totals[lbl] += 1

        if not records:
            raise ValueError('No labeled groups found; adjust filtering thresholds.')

        if verbose:
            import sys
            breakdown = ', '.join(f"{CLASS_NAMES.get(i, str(i))}: {int(label_totals[i])}" for i in range(NUM_GROUP_CLASSES))
            print(f"Loaded {len(records)} groups from {len(paths)} files ({breakdown})", file=sys.stderr, flush=True)

        return records


