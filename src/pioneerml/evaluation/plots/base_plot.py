from __future__ import annotations

import numpy as np
import torch


class BasePlot:
    """Base plot interface."""

    name: str = "base"

    def render(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.render(*args, **kwargs)


def _to_numpy(arr):
    if torch.is_tensor(arr):
        return arr.detach().cpu().numpy()
    if isinstance(arr, (list, tuple)):
        return np.asarray(arr)
    if isinstance(arr, np.ndarray):
        return arr
    raise TypeError(f"Unsupported input type for plotting: {type(arr)}")
