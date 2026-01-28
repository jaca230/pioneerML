from __future__ import annotations

from typing import Optional

import polars as pl

from pioneerml.data.processing.base import BaseProcessor
from pioneerml.data.processing.time_groups import assign_time_group_labels


class GroupClassificationProcessor(BaseProcessor):
    """Adds time-group labels; conversion happens in the dataset."""

    def __init__(
        self,
        *,
        max_hits: int = 256,
        pad_value: float = 0.0,
        compute_time_groups: bool = True,
        time_window_ns: float = 1.0,
    ):
        self.max_hits = max_hits
        self.pad_value = pad_value
        self.compute_time_groups = compute_time_groups
        self.time_window_ns = time_window_ns

    @property
    def columns(self):
        base_cols = [
            "event_id",
            "pion_in_group",
            "muon_in_group",
            "mip_in_group",
            "pion_stop_x",
            "pion_stop_y",
            "pion_stop_z",
            "total_pion_energy",
            "total_muon_energy",
            "total_mip_energy",
            "start_x",
            "start_y",
            "start_z",
            "end_x",
            "end_y",
            "end_z",
            "true_arc_length",
            "hits_x",
            "hits_y",
            "hits_z",
            "hits_edep",
            "hits_strip_type",
            "hits_pdg_id",
        ]
        if self.compute_time_groups:
            base_cols.append("hits_time")
        return base_cols

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.compute_time_groups and "hits_time_group" not in df.columns:
            df = df.with_columns(
                pl.col("hits_time").map_elements(
                    lambda times: assign_time_group_labels(times, self.time_window_ns), return_dtype=pl.List(pl.Int32)
                )
            )
        return df
