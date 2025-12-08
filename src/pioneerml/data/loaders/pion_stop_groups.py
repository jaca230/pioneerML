"""
Loader for pion stop position regression data.
"""

from typing import List, Dict, Any, Optional
import numpy as np

from .base import BaseLoader


class PionStopGroupsLoader(BaseLoader):
    """Loader for pion stop position regression data."""

    def load(
        self,
        file_pattern: str,
        *,
        pion_pdg: int = 1,
        max_files: Optional[int] = None,
        limit_groups: Optional[int] = None,
        min_hits: int = 3,
        min_pion_hits: int = 1,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Load preprocessed pion groups for stop position regression.
        
        Args:
            file_pattern: Glob pattern for .npy files
            pion_pdg: PDG code for pions (default 1)
            max_files: Maximum number of files to load
            limit_groups: Maximum number of groups to load
            min_hits: Minimum number of hits per group
            min_pion_hits: Minimum number of pion hits per group
            verbose: Whether to print loading statistics
            
        Returns:
            List of dictionaries with keys: coord, z, view, energy, time, pdg, event_id, group_id,
                true_x, true_y, true_z, true_time
        """
        paths = self._find_files(file_pattern, max_files=max_files, verbose=verbose)

        records: List[Dict[str, Any]] = []
        total_groups = 0
        kept_groups = 0
        total_pion_hits = 0
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

                    arr = np.asarray(raw_group)
                    if arr.ndim != 2 or arr.shape[0] < min_hits or arr.shape[1] < 12:
                        continue

                    pdgs = arr[:, 5].astype(int)

                    allowed_mask = (pdgs == pion_pdg)
                    if not np.all(allowed_mask):
                        continue

                    pion_mask = pdgs == pion_pdg
                    pion_hits = int(pion_mask.sum())
                    total_groups += 1
                    if pion_hits < min_pion_hits:
                        continue

                    kept_groups += 1
                    total_pion_hits += pion_hits

                    record_event_id = self._extract_event_id(arr, path, event_offset)

                    records.append({
                        "coord": arr[:, 0].astype(np.float32),
                        "z": arr[:, 1].astype(np.float32),
                        "view": arr[:, 2].astype(np.float32),
                        "energy": arr[:, 3].astype(np.float32),
                        "time": arr[:, 4].astype(np.float32),
                        "pdg": pdgs.astype(np.int32),
                        "event_id": record_event_id,
                        "group_id": int(group_idx),
                        "true_x": arr[:, 7].astype(np.float32),
                        "true_y": arr[:, 8].astype(np.float32),
                        "true_z": arr[:, 9].astype(np.float32),
                        "true_time": arr[:, 10].astype(np.float32),
                    })

        if verbose:
            print(
                f"Loaded {len(records)} pion groups across {len(paths)} files | "
                f"total groups={total_groups} kept={kept_groups} | total pion hits={total_pion_hits}"
            )

        return records

