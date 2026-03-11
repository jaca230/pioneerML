from .config.hpo_runtime_config_resolver import HPORuntimeConfigResolver
from .payload.hpo_runtime_state_resolver import HPORuntimeStateResolver


def resolve_batch_size_search(cfg, *, default_min_exp: int = 5, default_max_exp: int = 7):
    return HPORuntimeConfigResolver.resolve_batch_size_search(
        cfg=cfg,
        default_min_exp=default_min_exp,
        default_max_exp=default_max_exp,
    )

__all__ = ["HPORuntimeConfigResolver", "HPORuntimeStateResolver", "resolve_batch_size_search"]
