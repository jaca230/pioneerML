from __future__ import annotations

from collections.abc import Mapping


def suggest_range(cfg: Mapping, key: str, *, default_low: float, default_high: float):
    raw = cfg.get(key)
    if isinstance(raw, Mapping):
        low = float(raw.get("low", default_low))
        high = float(raw.get("high", default_high))
        log = bool(raw.get("log", True))
        return low, high, log
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return float(raw[0]), float(raw[1]), True
    return float(default_low), float(default_high), True


def resolve_batch_size_search(cfg: Mapping, *, default_min_exp: int = 5, default_max_exp: int = 7):
    raw = cfg.get("batch_size", {"min_exp": default_min_exp, "max_exp": default_max_exp})
    if isinstance(raw, Mapping):
        min_exp = int(raw.get("min_exp", default_min_exp))
        max_exp = int(raw.get("max_exp", default_max_exp))
        if min_exp > max_exp:
            min_exp, max_exp = max_exp, min_exp
        return None, min_exp, max_exp
    if isinstance(raw, (list, tuple)):
        values = [int(v) for v in raw if int(v) > 0]
        if not values:
            return 1, 0, 0
        if len(values) == 1:
            return values[0], 0, 0
        min_value = min(values)
        max_value = max(values)
        min_exp = int(max(min_value - 1, 0)).bit_length()
        max_exp = int(max_value).bit_length() - 1
        if min_exp > max_exp:
            return min_value, 0, 0
        return None, min_exp, max_exp
    fixed = int(raw)
    return fixed, 0, 0


def build_hpo_trainer_kwargs(cfg: Mapping) -> dict:
    kwargs = dict(cfg.get("trainer_kwargs") or {})
    max_train_batches = cfg.get("max_train_batches")
    if max_train_batches is not None:
        kwargs.setdefault("limit_train_batches", int(max_train_batches))
    max_val_batches = cfg.get("max_val_batches")
    if max_val_batches is not None:
        kwargs.setdefault("limit_val_batches", int(max_val_batches))
    return kwargs
