"""
Utilities for assigning time-group labels to hits.

The grouping logic mirrors the legacy `group_hits_in_time` from
deprecated/omar_pioneerML: hits are sorted by time and a new group
starts whenever the gap to the previous hit exceeds a configurable
window.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Any

import numpy as np
import polars as pl


def assign_time_group_labels(times: Sequence[float] | None, window_ns: float = 1.0) -> list[int]:
    """
    Assign incremental time-group labels to a list of hit times.

    Args:
        times: Iterable of hit times (ns).
        window_ns: Maximum allowed gap (ns) between consecutive hits in a group.

    Returns:
        List of group ids (0-based) aligned with the input order.
    """
    if times is None:
        return []
    arr = np.asarray(list(times), dtype=np.float64)
    if arr.size == 0:
        return []

    order = np.argsort(arr)
    sorted_t = arr[order]
    # A new group starts when the gap exceeds the window (strictly greater, matching legacy code).
    new_group_flags = np.diff(sorted_t) > window_ns
    group_ids_sorted = np.cumsum(np.concatenate([[0], new_group_flags.astype(np.int64)]))

    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(order.size)
    group_ids = group_ids_sorted[inv_order]
    return group_ids.tolist()


def add_time_group_labels(
    df: pl.DataFrame,
    *,
    time_col: str = "hits_time",
    output_col: str = "hits_time_group",
    window_ns: float = 1.0,
) -> pl.DataFrame:
    """
    Add a list column of time-group labels to a Parquet DataFrame.

    Args:
        df: Polars DataFrame with a list column of hit times.
        time_col: Name of the list column containing hit times.
        output_col: Name of the output list column with group labels.
        window_ns: Maximum allowed gap (ns) between consecutive hits in a group.

    Returns:
        DataFrame with the new list column `output_col`.
    """
    if time_col not in df.columns:
        raise KeyError(f"{time_col} not found in DataFrame")

    return df.with_columns(
        pl.col(time_col)
        .map_elements(lambda times: assign_time_group_labels(times, window_ns), return_dtype=pl.List(pl.Int32))
        .alias(output_col)
    )


__all__ = ["assign_time_group_labels", "add_time_group_labels"]
