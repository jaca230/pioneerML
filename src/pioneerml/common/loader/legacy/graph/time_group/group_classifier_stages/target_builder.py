from __future__ import annotations

import numpy as np
import torch


class TargetBuilder:
    """Build graph-level training targets for group-classifier."""

    def build(
        self,
        *,
        loader,
        include_targets: bool,
        chunk_in: dict[str, np.ndarray],
        total_graphs: int,
        local_gid: np.ndarray,
        row_ids_graph: np.ndarray,
    ) -> torch.Tensor | None:
        if not include_targets:
            return None

        y_out = np.zeros((int(total_graphs), loader.NUM_CLASSES), dtype=np.float32)
        loader._populate_target_labels(
            y_out=y_out,
            total_graphs=int(total_graphs),
            local_gid=local_gid,
            row_ids_graph=row_ids_graph,
            group_pion_in_values=chunk_in["group_pion_in_values"],
            group_pion_in_offsets=chunk_in["group_pion_in_offsets"],
            group_muon_in_values=chunk_in["group_muon_in_values"],
            group_muon_in_offsets=chunk_in["group_muon_in_offsets"],
            group_mip_in_values=chunk_in["group_mip_in_values"],
            group_mip_in_offsets=chunk_in["group_mip_in_offsets"],
        )
        return torch.from_numpy(y_out)
