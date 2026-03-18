from .config.hpo_runtime_config_resolver import HPORuntimeConfigResolver


def resolve_batch_size_search(cfg, *, default_min_exp: int = 5, default_max_exp: int = 7):
    return HPORuntimeConfigResolver.resolve_batch_size_search(
        cfg=cfg,
        default_min_exp=default_min_exp,
        default_max_exp=default_max_exp,
    )

__all__ = ["HPORuntimeConfigResolver", "resolve_batch_size_search"]
