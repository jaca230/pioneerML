"""
Loader for positron angle regression data.

TODO: Angle data is not currently stored in mainTimeGroups files.
      This loader requires angle_targets_pattern to load angles from separate files.
      Once angle data is added to mainTimeGroups, this loader should be updated to extract
      angles directly from the group arrays.
"""

import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from .base import BaseLoader


class PositronAngleGroupsLoader(BaseLoader):
    """
    Loader for positron angle regression data.
    
    TODO: Currently requires separate angle target files. Once angles are added to
          mainTimeGroups, update to extract directly from group arrays.
    """

    def load(
        self,
        file_pattern: str,
        *,
        max_files: Optional[int] = None,
        limit_groups: Optional[int] = None,
        min_hits: int = 2,
        angle_targets_pattern: Optional[str] = None,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Load groups for positron angle regression.

        TODO: Angle data is NOT currently in mainTimeGroups files.
              This loader requires angle_targets_pattern to load angles from separate files.
              Once angles are added to mainTimeGroups, this should be updated to extract
              angles directly from the group arrays.
        
        Angles are loaded from separate files via angle_targets_pattern and matched to
        groups by index (same order). Each angle target should be [theta, phi] in radians
        or a unit vector [x, y, z].
        
        Args:
            file_pattern: Glob pattern for .npy files containing groups
            max_files: Maximum number of files to load
            limit_groups: Maximum number of groups to load
            min_hits: Minimum number of hits per group
            angle_targets_pattern: REQUIRED - Glob pattern for separate angle target files.
                Angles are matched to groups by index (same order).
            verbose: Whether to print loading statistics
            
        Returns:
            List of dictionaries with keys: coord, z, energy, view, angle, event_id, group_id
            
        Raises:
            ValueError: If angle_targets_pattern is not provided or if there are not enough
                angle targets for the number of groups loaded.
        """
        angle_targets: List[np.ndarray] | None = None
        if angle_targets_pattern is not None:
            angle_paths = sorted(Path(p) for p in glob.glob(angle_targets_pattern))
            if not angle_paths:
                raise FileNotFoundError(f"No angle target files matched pattern '{angle_targets_pattern}'")
            angle_targets = []
            for p in angle_paths:
                arr = np.load(p, allow_pickle=True)
                # Flatten list of per-file targets
                if arr.ndim == 1:
                    angle_targets.extend([np.asarray(x, dtype=np.float32).reshape(-1) for x in arr])
                else:
                    angle_targets.extend([np.asarray(x, dtype=np.float32).reshape(-1) for x in arr.reshape(-1, arr.shape[-1])])

        paths = self._find_files(file_pattern, max_files=max_files, verbose=verbose)

        records: List[Dict[str, Any]] = []
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
                    if group_arr.ndim != 2 or group_arr.shape[0] < min_hits or group_arr.shape[1] < 4:
                        continue

                    record_event_id = self._extract_event_id(group_arr, path, event_offset)

                    # TODO: Angle data is not currently in mainTimeGroups files.
                    #       Once angles are added to the data format, extract them here.
                    #       For now, angles must come from separate files via angle_targets_pattern.
                    
                    if angle_targets is None:
                        raise ValueError(
                            "angle_targets_pattern is REQUIRED. "
                            "Angle data is not currently stored in mainTimeGroups files. "
                            "Provide a pattern to load angle targets from separate files. "
                            "TODO: Update this loader once angles are added to mainTimeGroups."
                        )
                    
                    if len(angle_targets) <= len(records):
                        raise ValueError(
                            f"Not enough angle targets provided. "
                            f"Loaded {len(records)} groups but only {len(angle_targets)} angle targets available."
                        )
                    
                    # Match angle target by index (same order as groups)
                    angle = np.asarray(angle_targets[len(records)], dtype=np.float32).reshape(-1)
                    
                    # Ensure angle has at least 2 components (theta, phi)
                    if angle.size < 2:
                        raise ValueError(
                            f"Angle target has only {angle.size} components, "
                            f"expected at least 2 (theta, phi)"
                        )

                    records.append(
                        {
                            "coord": group_arr[:, 0].astype(np.float32),
                            "z": group_arr[:, 1].astype(np.float32),
                            "energy": group_arr[:, 3].astype(np.float32),
                            "view": group_arr[:, 2].astype(np.float32),
                            "angle": angle[:2],
                            "event_id": record_event_id,
                            "group_id": group_idx,
                        }
                    )

        if not records:
            raise ValueError("No groups found; adjust filtering thresholds or file pattern.")

        if verbose:
            import sys
            print(f"Loaded {len(records)} groups from {len(paths)} files with angle targets", file=sys.stderr, flush=True)

        return records

