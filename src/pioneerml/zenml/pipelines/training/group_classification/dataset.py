from __future__ import annotations

from typing import Optional

import polars as pl
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from pioneerml.data.datasets.utils import fully_connected_edge_index, build_edge_attr
from pioneerml.data.processing.time_groups import assign_time_group_labels


class GroupClassificationPolarsDataset(Dataset):
    """
    Dataset that keeps parquet data columnar (Polars) and materializes tensors per sample.
    Only one copy occurs when reading parquet; per-sample tensors are built directly from column slices.
    """

    def __init__(
        self,
        frame: pl.DataFrame,
        *,
        max_hits: int = 256,
        pad_value: float = 0.0,
        compute_time_groups: bool = True,
        time_window_ns: float = 1.0,
    ):
        if compute_time_groups and "hits_time_group" not in frame.columns:
            frame = frame.with_columns(
                pl.col("hits_time").map_elements(
                    lambda times: assign_time_group_labels(times, time_window_ns), return_dtype=pl.List(pl.Int32)
                )
            )

        self.frame = frame
        self.max_hits = max_hits
        self.pad_value = pad_value
        self.compute_time_groups = compute_time_groups

        # Cache column references for quick access
        self._cols = {c: frame[c] for c in frame.columns}

    def __len__(self) -> int:
        return self.frame.height

    def __getitem__(self, idx: int) -> Data:
        row = {name: col[idx] for name, col in self._cols.items()}

        hits_z = row.get("hits_z") or []
        hits_x = row.get("hits_x") or []
        hits_y = row.get("hits_y") or []
        hits_edep = row.get("hits_edep") or []
        hits_view = row.get("hits_strip_type") or []
        hits_pdg = row.get("hits_pdg_id") or []
        hits_tg = row.get("hits_time_group") if self.compute_time_groups else None

        num_hits = len(hits_z)
        valid_len = min(num_hits, self.max_hits)
        pad_len = self.max_hits - valid_len

        # Build node feature tensor
        x = torch.zeros((self.max_hits, 4), dtype=torch.float32)
        view_tensor = torch.tensor([0 if v is None else float(v) for v in hits_view[:valid_len]], dtype=torch.float32)
        z_tensor = torch.tensor(hits_z[:valid_len], dtype=torch.float32)
        e_tensor = torch.tensor(hits_edep[:valid_len], dtype=torch.float32)
        coords = []
        for i in range(valid_len):
            v = int(view_tensor[i].item())
            vx = hits_x[i] if i < len(hits_x) and hits_x[i] is not None else self.pad_value
            vy = hits_y[i] if i < len(hits_y) and hits_y[i] is not None else self.pad_value
            coords.append(float(vx if v == 0 else vy))
        x[:valid_len, 0] = torch.tensor(coords, dtype=torch.float32)
        x[:valid_len, 1] = z_tensor
        x[:valid_len, 2] = e_tensor
        x[:valid_len, 3] = view_tensor
        if pad_len:
            x[valid_len:, 3] = -1.0  # sentinel view

        hit_mask = torch.zeros(self.max_hits, dtype=torch.bool)
        hit_mask[:valid_len] = True

        tg_tensor = None
        if hits_tg is not None:
            tg_vals = list(hits_tg)[:valid_len]
            if pad_len:
                tg_vals += [-1] * pad_len
            tg_tensor = torch.tensor(tg_vals, dtype=torch.long)

        # Build graph components using only valid hits
        edge_index = fully_connected_edge_index(valid_len, device=x.device)
        edge_attr = build_edge_attr(x[:valid_len], edge_index)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.hit_mask = hit_mask
        data.num_valid_hits = torch.tensor([valid_len], dtype=torch.long)
        if tg_tensor is not None:
            data.time_group_ids = tg_tensor

        # Labels and globals
        labels = []
        if int(row.get("pion_in_group", 0)):
            labels.append(0)
        if int(row.get("muon_in_group", 0)):
            labels.append(1)
        if int(row.get("mip_in_group", 0)):
            labels.append(2)

        y = torch.zeros(3, dtype=torch.float32)
        for lbl in labels:
            if 0 <= lbl < 3:
                y[lbl] = 1.0
        data.y = y

        data.u = torch.tensor([[x[:valid_len, 2].sum()]], dtype=torch.float32)

        data.event_id = torch.tensor(int(row.get("event_id", -1)), dtype=torch.long)
        data.group_id = torch.tensor(int(idx), dtype=torch.long)

        class_energies = [
            float(row.get("total_pion_energy", 0.0)),
            float(row.get("total_muon_energy", 0.0)),
            float(row.get("total_mip_energy", 0.0)),
        ]
        data.y_energy = torch.tensor(class_energies, dtype=torch.float32).unsqueeze(0)

        true_start = [float(row.get("start_x", 0.0)), float(row.get("start_y", 0.0)), float(row.get("start_z", 0.0))]
        true_end = [float(row.get("end_x", 0.0)), float(row.get("end_y", 0.0)), float(row.get("end_z", 0.0))]
        data.y_pos = torch.tensor([true_start, true_end], dtype=torch.float32).unsqueeze(0)

        pion_stop = [
            float(row.get("pion_stop_x", 0.0)),
            float(row.get("pion_stop_y", 0.0)),
            float(row.get("pion_stop_z", 0.0)),
        ]
        data.y_pion_stop = torch.tensor(pion_stop, dtype=torch.float32).unsqueeze(0)

        data.y_arc = torch.tensor([float(row.get("true_arc_length", 0.0))], dtype=torch.float32).unsqueeze(0)

        hit_pdgs = [int(v) if v is not None else -1 for v in hits_pdg[:valid_len]]
        if pad_len:
            hit_pdgs += [-1] * pad_len
        data.y_node = torch.tensor(hit_pdgs, dtype=torch.long)

        return data
