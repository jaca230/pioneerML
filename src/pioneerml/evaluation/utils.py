"""
Helpers for pulling or computing predictions/targets from dict-like inputs.
"""

from __future__ import annotations

from typing import Any, Tuple

import torch


def resolve_preds_targets(container: dict[str, Any], dataloader: str = "val") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Retrieve predictions and targets from a dict-like container, or compute them using a module/datamodule.

    The function looks for:
    - cached ``preds`` / ``targets`` in the container
    - a Lightning/PyTorch ``module`` or ``lightning_module`` and ``datamodule`` to generate them
    """
    getter = container.get
    preds = getter("preds")
    targets = getter("targets")
    if preds is not None and targets is not None:
        return preds, targets

    module = getter("lightning_module") or getter("module")
    datamodule = getter("datamodule")
    if module is None or datamodule is None:
        raise RuntimeError("No predictions/targets available and no module/datamodule to compute them. Run training first.")

    module = module.eval()
    loader_fn = getattr(datamodule, f"{dataloader}_dataloader", None)
    if loader_fn is None:
        raise RuntimeError(f"Datamodule has no {dataloader}_dataloader")

    dl = loader_fn()
    if isinstance(dl, list):
        dl = dl[0]

    preds_list, targets_list = [], []
    device = getattr(module, "device", None)
    with torch.no_grad():
        for batch in dl:
            batch = batch.to(device) if device is not None else batch
            out = module(batch)
            preds_list.append(out.detach().cpu())
            targets_list.append(batch.y.detach().cpu())

    preds = torch.cat(preds_list)
    targets = torch.cat(targets_list)

    # cache for reuse
    try:
        container["preds"] = preds
        container["targets"] = targets
    except Exception:
        pass
    return preds, targets
