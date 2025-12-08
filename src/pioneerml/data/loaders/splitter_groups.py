"""
Loader for splitter groups (per-hit classification).
"""

from typing import List, Dict, Any, Optional
import numpy as np

from .base import BaseLoader
from .constants import PION_MASK, MUON_MASK, POSITRON_MASK, ELECTRON_MASK


def _multilabel_from_mask(mask: int) -> Optional[List[int]]:
    """Return [is_pion, is_muon, is_mip] as ints 0/1, or None to drop hit."""
    has_pion = bool(mask & PION_MASK)
    has_muon = bool(mask & MUON_MASK)
    has_pos = bool(mask & POSITRON_MASK)
    has_ele = bool(mask & ELECTRON_MASK)

    has_mip = has_pos or has_ele

    # Drop pure OTHER hits: no pion/muon/e± content at all
    if not (has_pion or has_muon or has_mip):
        return None

    return [
        int(has_pion),
        int(has_muon),
        int(has_mip),
    ]


def is_multispecies_group(node_labels: np.ndarray) -> bool:
    """
    Check if a group contains >= 2 particle species.
    
    Args:
        node_labels: [N, 3] array of 0/1 for [pion, muon, mip]
        
    Returns:
        True if the group contains >= 2 particle species
    """
    species_present = node_labels.sum(axis=0) > 0  # boolean array of shape [3]
    num_species = species_present.sum()
    return num_species >= 2


class SplitterGroupsLoader(BaseLoader):
    """Loader for splitter groups (per-hit classification)."""

    def load(
        self,
        file_pattern: str,
        *,
        max_files: Optional[int] = None,
        limit_groups: Optional[int] = None,
        min_hits: int = 3,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Load groups for the splitter network (per-hit classification).
        
        Args:
            file_pattern: Glob pattern for .npy files
            max_files: Maximum number of files to load
            limit_groups: Maximum number of groups to load
            min_hits: Minimum number of hits per group
            verbose: Whether to print loading statistics
            
        Returns:
            List of dictionaries with keys: coord, z, energy, view, hit_labels, event_id, group_id
        """
        paths = self._find_files(file_pattern, max_files=max_files, verbose=verbose)

        records: List[Dict[str, Any]] = []
        # per-channel totals: [pion_hits, muon_hits, mip_hits]
        label_totals = np.zeros(3, dtype=int)

        for path in paths:
            if limit_groups is not None and len(records) >= limit_groups:
                break
            chunk = np.load(path, allow_pickle=True)
            chunk_index = self._extract_file_index(path) or 0

            for event_offset, event_groups in enumerate(chunk):
                if limit_groups is not None and len(records) >= limit_groups:
                    break
                if event_groups is None or len(event_groups) == 0:
                    continue

                for group_idx, raw_group in enumerate(event_groups):
                    if limit_groups is not None and len(records) >= limit_groups:
                        break
                    group = np.asarray(raw_group)
                    if group.ndim != 2 or group.shape[0] < min_hits or group.shape[1] < 6:
                        continue

                    mask_values = group[:, 5].astype(int)
                    node_labels: List[List[int]] = []
                    keep_indices: List[int] = []

                    for hit_idx, mask in enumerate(mask_values):
                        bits = _multilabel_from_mask(mask)
                        if bits is None:
                            continue  # pure OTHER
                        keep_indices.append(hit_idx)
                        node_labels.append(bits)
                        label_totals += np.array(bits, dtype=int)

                    if len(node_labels) < 1:
                        continue  # need at least one labelled hit to form a graph

                    if not is_multispecies_group(np.array(node_labels)):  # skip single-species groups for training
                        continue

                    filtered = group[keep_indices]
                    coord = filtered[:, 0].astype(np.float32)
                    z_pos = filtered[:, 1].astype(np.float32)
                    view_flag = filtered[:, 2].astype(np.float32)
                    energy = filtered[:, 3].astype(np.float32)

                    record_event_id: Optional[int] = None
                    if filtered.shape[1] > 6:
                        try:
                            record_event_id = int(filtered[0, 6])
                        except (ValueError, TypeError):
                            record_event_id = None
                    if record_event_id is None:
                        record_event_id = chunk_index * 100000 + event_offset

                    records.append({
                        'coord': coord,
                        'z': z_pos,
                        'view': view_flag,
                        'energy': energy,
                        # shape [num_hits, 3]
                        'hit_labels': node_labels,
                        'event_id': record_event_id,
                        'group_id': len(records),
                    })

        if not records:
            raise ValueError('No labelled splitter groups found; adjust filtering thresholds.')

        if verbose:
            total_hits = label_totals.sum()
            pion_hits, muon_hits, mip_hits = label_totals.tolist()
            print(
                f"Loaded {len(records)} splitter groups from {len(paths)} files; "
                f"pion_hits={pion_hits}, muon_hits={muon_hits}, mip_hits={mip_hits}, total_hits={total_hits}"
            )

        return records

