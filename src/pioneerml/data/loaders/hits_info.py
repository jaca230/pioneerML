"""
Loader for paired hits/group-info NPY dumps produced by root_to_npy_converter.

Supports both Dec 12 format (no arc length) and Dec 31 format (includes arc length).
Missing arc length is filled with 0 to stay forward-compatible.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from pioneerml.data.datasets.graph_group import GraphRecord
from pioneerml.data.loaders.base import BaseLoader
from pioneerml.data.loaders.constants import (
    BIT_TO_CLASS,
    PION_MASK,
    MUON_MASK,
    POSITRON_MASK,
    ELECTRON_MASK,
    CLASS_NAMES,
)


def _mask_to_multilabel(mask: int) -> List[int]:
    """Return [is_pion, is_muon, is_mip] from a bitmask."""
    return [
        int(bool(mask & PION_MASK)),
        int(bool(mask & MUON_MASK)),
        int(bool(mask & (POSITRON_MASK | ELECTRON_MASK))),
    ]


def _mask_to_class(mask: int) -> int:
    """Collapse mask to a single class index, preferring pion>muon>mip."""
    for bit, cls in BIT_TO_CLASS.items():
        if mask & bit:
            return cls
    return -1


class HitsAndInfoLoader(BaseLoader):
    """
    Load paired hits/info NPY batches into GraphRecord objects.

    Expects matching file patterns for hits and group_info arrays:
    - hits rows: [coord, z, stripType(view), energy, pdg_mask]
    - info rows: [pion, muon, mip, pionStopX/Y/Z, E_pi/E_mu/E_mip, theta, phi, eventID,
                  startX/Y/Z, endX/Y/Z, (optional) true_arc_length]
    """

    def load(
        self,
        hits_pattern: str,
        info_pattern: str,
        *,
        max_files: Optional[int] = None,
        limit_groups: Optional[int] = None,
        min_hits: int = 2,
        verbose: bool = True,
        include_hit_labels: bool = False,
    ) -> List[GraphRecord]:
        hits_paths = self._find_files(hits_pattern, max_files=max_files, verbose=verbose)
        info_paths = self._find_files(info_pattern, max_files=max_files, verbose=verbose)

        if len(hits_paths) != len(info_paths):
            raise ValueError(f"Hits files ({len(hits_paths)}) and info files ({len(info_paths)}) count mismatch.")

        records: List[GraphRecord] = []
        for h_path, i_path in zip(hits_paths, info_paths):
            if limit_groups is not None and len(records) >= limit_groups:
                break

            hits_batch = np.load(h_path, allow_pickle=True)
            info_batch = np.load(i_path, allow_pickle=True)

            if len(hits_batch) != len(info_batch):
                raise ValueError(
                    f"Batch length mismatch: {h_path.name} hits={len(hits_batch)} info={len(info_batch)}"
                )

            for group_hits, group_info in zip(hits_batch, info_batch):
                if limit_groups is not None and len(records) >= limit_groups:
                    break

                hits_arr = np.asarray(group_hits)
                info_arr = np.asarray(group_info, dtype=np.float32)

                if hits_arr.ndim != 2 or hits_arr.shape[1] < 5 or hits_arr.shape[0] < min_hits:
                    continue

                coord = hits_arr[:, 0].astype(np.float32)
                z_pos = hits_arr[:, 1].astype(np.float32)
                view = hits_arr[:, 2].astype(np.float32)
                energy = hits_arr[:, 3].astype(np.float32)
                pdg_masks = hits_arr[:, 4].astype(int)

                labels = [idx for idx, flag in enumerate(info_arr[:3].astype(int)) if flag > 0]

                class_energies = info_arr[6:9] if info_arr.size >= 9 else None

                # True start/end and arc length (Dec 31 adds arc length at the end)
                true_start = info_arr[12:15] if info_arr.size >= 15 else None
                true_end = info_arr[15:18] if info_arr.size >= 18 else None
                true_arc_length = float(info_arr[17]) if info_arr.size >= 18 else 0.0

                pion_stop = info_arr[3:6] if info_arr.size >= 6 else None
                theta = float(info_arr[9]) if info_arr.size >= 10 else -1000.0
                phi = float(info_arr[10]) if info_arr.size >= 11 else -1000.0
                event_id = int(info_arr[11]) if info_arr.size >= 12 else -1

                angle_vec = None
                if theta != -1000 and phi != -1000:
                    sin_t = math.sin(theta)
                    cos_t = math.cos(theta)
                    sin_p = math.sin(phi)
                    cos_p = math.cos(phi)
                    angle_vec = [sin_t * cos_p, sin_t * sin_p, cos_t]

                hit_labels = None
                if include_hit_labels:
                    hit_labels = [_mask_to_multilabel(int(mask)) for mask in pdg_masks]

                hit_pdgs = [_mask_to_class(int(mask)) for mask in pdg_masks]

                records.append(
                    GraphRecord(
                        coord=coord,
                        z=z_pos,
                        energy=energy,
                        view=view,
                        labels=labels,
                        event_id=event_id,
                        group_id=len(records),
                        hit_labels=hit_labels,
                        hit_pdgs=hit_pdgs,
                        class_energies=class_energies,
                        true_start=true_start,
                        true_end=true_end,
                        true_pion_stop=pion_stop if pion_stop is not None else None,
                        true_angle_vector=angle_vec,
                        true_arc_length=true_arc_length,
                    )
                )

        if verbose:
            import sys

            print(f"Loaded {len(records)} groups from {len(hits_paths)} file pairs", file=sys.stderr, flush=True)

        return records
