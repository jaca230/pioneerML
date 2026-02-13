from __future__ import annotations

import numpy as np

from pioneerml.common.pipeline_utils.derivers.time_grouper import TimeGrouper


class GroupClassifierSummaryDeriver:
    """Minimal deriver for group-classification labels and time groups.

    Outputs only what group-classifier loaders need:
    - hits_time_group
    - pion_in_group
    - muon_in_group
    - mip_in_group
    """

    def __init__(self, window_ns: float = 1.0):
        self.time_grouper = TimeGrouper(window_ns=float(window_ns))

    def derive_row(self, row: dict) -> dict:
        hits_time = np.asarray(row.get("hits_time") or [], dtype=np.float64)
        n_hits = int(hits_time.shape[0])
        hit_groups = self.time_grouper.derive(hits_time)
        num_groups = int(hit_groups.max()) + 1 if n_hits > 0 else 0

        steps_mc = row.get("steps_mc_event_id") or []
        steps_step = row.get("steps_step_id") or []
        steps_pdg = row.get("steps_pdg_id") or []
        m = min(len(steps_mc), len(steps_step), len(steps_pdg))

        step_class: dict[tuple[int, int], int] = {}
        for i in range(m):
            cls = self._class_from_pdg(int(steps_pdg[i]))
            if cls < 0:
                continue
            key = (int(steps_mc[i]), int(steps_step[i]))
            if key not in step_class:
                step_class[key] = cls

        pion = [0] * num_groups
        muon = [0] * num_groups
        mip = [0] * num_groups

        contrib_mc = row.get("hits_contrib_mc_event_id") or [[] for _ in range(n_hits)]
        contrib_step = row.get("hits_contrib_step_id") or [[] for _ in range(n_hits)]
        if len(contrib_mc) != n_hits or len(contrib_step) != n_hits:
            raise ValueError("Contribution arrays must match hit count.")

        for i in range(n_hits):
            gid = int(hit_groups[i])
            mc_ids = contrib_mc[i] or []
            step_ids = contrib_step[i] or []
            for j in range(min(len(mc_ids), len(step_ids))):
                cls = step_class.get((int(mc_ids[j]), int(step_ids[j])), -1)
                if cls == 0:
                    pion[gid] = 1
                elif cls == 1:
                    muon[gid] = 1
                elif cls == 2:
                    mip[gid] = 1

        return {
            "hits_time_group": hit_groups.tolist(),
            "pion_in_group": pion,
            "muon_in_group": muon,
            "mip_in_group": mip,
        }

    def _class_from_pdg(self, pdg: int) -> int:
        if pdg == 211:
            return 0
        if pdg == -13:
            return 1
        if pdg in (-11, 11):
            return 2
        return -1
