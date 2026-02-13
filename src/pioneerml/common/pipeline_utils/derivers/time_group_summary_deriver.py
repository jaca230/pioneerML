from __future__ import annotations

import math

import numpy as np

from pioneerml.common.pipeline_utils.derivers.base_deriver import BaseDeriver
from pioneerml.common.pipeline_utils.derivers.particle_mask_deriver import ParticleMaskDeriver
from pioneerml.common.pipeline_utils.derivers.time_grouper import TimeGrouper


class TimeGroupSummaryDeriver(BaseDeriver):
    """Derive time-group ids, class labels, per-group energies, and endpoints."""

    def __init__(self, window_ns: float = 1.0):
        self.time_grouper = TimeGrouper(window_ns=float(window_ns))
        self.mask_deriver = ParticleMaskDeriver()

    def derive_row(self, row: dict) -> dict:
        hits_time = np.asarray(row.get("hits_time") or [], dtype=np.float64)
        hits_edep = np.asarray(row.get("hits_edep") or [], dtype=np.float64)
        n_hits = int(hits_time.shape[0])
        if hits_edep.shape[0] != n_hits:
            raise ValueError("hits_time and hits_edep lengths must match.")

        hit_groups = self.time_grouper.derive(hits_time)
        num_groups = int(hit_groups.max()) + 1 if n_hits > 0 else 0

        steps_mc = row.get("steps_mc_event_id") or []
        steps_step = row.get("steps_step_id") or []
        steps_pdg = row.get("steps_pdg_id") or []
        steps_x = row.get("steps_x") or []
        steps_y = row.get("steps_y") or []
        steps_z = row.get("steps_z") or []
        steps_edep = row.get("steps_edep") or []
        steps_time = row.get("steps_time") or []

        m = min(
            len(steps_mc),
            len(steps_step),
            len(steps_pdg),
            len(steps_x),
            len(steps_y),
            len(steps_z),
            len(steps_edep),
            len(steps_time),
        )
        step_truth: dict[tuple[int, int], tuple[int, float, float, float, float, float]] = {}
        for i in range(m):
            key = (int(steps_mc[i]), int(steps_step[i]))
            if key not in step_truth:
                step_truth[key] = (
                    int(steps_pdg[i]),
                    float(steps_x[i]),
                    float(steps_y[i]),
                    float(steps_z[i]),
                    float(steps_time[i]),
                    max(0.0, float(steps_edep[i])),
                )

        contrib_mc = row.get("hits_contrib_mc_event_id") or [[] for _ in range(n_hits)]
        contrib_step = row.get("hits_contrib_step_id") or [[] for _ in range(n_hits)]
        if len(contrib_mc) != n_hits or len(contrib_step) != n_hits:
            raise ValueError("Contribution arrays must match hit count.")

        fallback_pdgs = row.get("hits_pdg_id")
        hit_pdgs: list[int] = [0] * n_hits
        hit_masks: list[int] = [self.mask_deriver.mask_from_pdg(0)] * n_hits

        group_has_pion = [0] * num_groups
        group_has_muon = [0] * num_groups
        group_has_mip = [0] * num_groups
        pion_energy = [0.0] * num_groups
        muon_energy = [0.0] * num_groups
        mip_energy = [0.0] * num_groups

        # endpoints/arc from unique true step points per group
        group_all_points: list[dict[tuple[int, int], tuple[float, float, float, float, int]]] = [
            {} for _ in range(num_groups)
        ]
        group_non_e_points: list[dict[tuple[int, int], tuple[float, float, float, float, int]]] = [
            {} for _ in range(num_groups)
        ]

        for i in range(n_hits):
            gid = int(hit_groups[i]) if n_hits else 0
            pdg_energy: dict[int, float] = {}
            mc_ids = contrib_mc[i] or []
            step_ids = contrib_step[i] or []
            for j in range(min(len(mc_ids), len(step_ids))):
                key = (int(mc_ids[j]), int(step_ids[j]))
                if key not in step_truth:
                    continue
                pdg, sx, sy, sz, st, sedep = step_truth[key]
                pdg_energy[pdg] = pdg_energy.get(pdg, 0.0) + sedep
                if 0 <= gid < num_groups:
                    if key not in group_all_points[gid]:
                        group_all_points[gid][key] = (sx, sy, sz, st, pdg)
                    if pdg not in (11, -11) and key not in group_non_e_points[gid]:
                        group_non_e_points[gid][key] = (sx, sy, sz, st, pdg)

            selected_pdg = 0
            if pdg_energy:
                selected_pdg = min(
                    pdg_energy.items(),
                    key=lambda kv: (-kv[1], kv[0]),
                )[0]
            elif fallback_pdgs is not None and i < len(fallback_pdgs):
                selected_pdg = int(fallback_pdgs[i])

            hit_pdgs[i] = int(selected_pdg)
            hit_masks[i] = int(self.mask_deriver.mask_from_pdg(selected_pdg))

            cls = self._class_from_pdg(selected_pdg)
            if 0 <= gid < num_groups:
                e = float(hits_edep[i])
                if cls == 0:
                    group_has_pion[gid] = 1
                    pion_energy[gid] += e
                elif cls == 1:
                    group_has_muon[gid] = 1
                    muon_energy[gid] += e
                elif cls == 2:
                    group_has_mip[gid] = 1
                    mip_energy[gid] += e

        group_start_x = [0.0] * num_groups
        group_start_y = [0.0] * num_groups
        group_start_z = [0.0] * num_groups
        group_end_x = [0.0] * num_groups
        group_end_y = [0.0] * num_groups
        group_end_z = [0.0] * num_groups
        group_true_arc_length = [0.0] * num_groups

        for gid in range(num_groups):
            points_map = group_non_e_points[gid] if group_non_e_points[gid] else group_all_points[gid]
            if not points_map:
                continue
            points = sorted(points_map.values(), key=lambda p: p[3])
            sx, sy, sz, _, _ = points[0]
            ex, ey, ez, _, _ = points[-1]
            group_start_x[gid] = float(sx)
            group_start_y[gid] = float(sy)
            group_start_z[gid] = float(sz)
            group_end_x[gid] = float(ex)
            group_end_y[gid] = float(ey)
            group_end_z[gid] = float(ez)

            arc = 0.0
            for i in range(1, len(points)):
                dx = points[i][0] - points[i - 1][0]
                dy = points[i][1] - points[i - 1][1]
                dz = points[i][2] - points[i - 1][2]
                arc += math.sqrt(dx * dx + dy * dy + dz * dz)
            group_true_arc_length[gid] = float(arc)

        return {
            "hits_time_group": hit_groups.tolist(),
            "hits_pdg_id": hit_pdgs,
            "hits_particle_mask": hit_masks,
            "pion_in_group": group_has_pion,
            "muon_in_group": group_has_muon,
            "mip_in_group": group_has_mip,
            "group_start_x": group_start_x,
            "group_start_y": group_start_y,
            "group_start_z": group_start_z,
            "group_end_x": group_end_x,
            "group_end_y": group_end_y,
            "group_end_z": group_end_z,
            "group_true_arc_length": group_true_arc_length,
            "pion_energy_per_group": pion_energy,
            "muon_energy_per_group": muon_energy,
            "mip_energy_per_group": mip_energy,
        }

    def _class_from_pdg(self, pdg: int) -> int:
        if pdg == 211:
            return 0
        if pdg == -13:
            return 1
        if pdg in (-11, 11):
            return 2
        return -1
