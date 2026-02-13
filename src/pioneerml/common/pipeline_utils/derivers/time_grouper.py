from __future__ import annotations

import numpy as np


class TimeGrouper:
    """Assign hit time-group ids using a fixed delta threshold."""

    def __init__(self, window_ns: float = 1.0):
        self.window_ns = float(window_ns)

    def derive(self, hit_times: np.ndarray) -> np.ndarray:
        n = int(hit_times.shape[0])
        if n == 0:
            return np.zeros((0,), dtype=np.int64)

        order = np.argsort(hit_times, kind="mergesort")
        sorted_times = hit_times[order]
        sorted_groups = np.zeros((n,), dtype=np.int64)
        gid = 0
        for i in range(1, n):
            if float(abs(sorted_times[i] - sorted_times[i - 1])) > self.window_ns:
                gid += 1
            sorted_groups[i] = gid

        groups = np.empty((n,), dtype=np.int64)
        groups[order] = sorted_groups
        return groups
