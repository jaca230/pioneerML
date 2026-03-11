from __future__ import annotations

from pioneerml.common.data_loader import LoaderFactory


def build_evaluation_loader_bundle(*, loader_factory: LoaderFactory, cfg: dict):
    params = LoaderFactory._resolve_loader_params(cfg, purpose="evaluate")
    raw_loader_cfg = cfg.get("loader_config")
    if isinstance(raw_loader_cfg, dict):
        if not isinstance(raw_loader_cfg.get("evaluate"), dict) and isinstance(raw_loader_cfg.get("val"), dict):
            params = LoaderFactory._resolve_loader_params(cfg, purpose="val")
    provider = loader_factory.build_loader(loader_params=params)
    loader = provider.make_dataloader(shuffle_batches=bool(params.get("shuffle_batches", False)))
    return provider, params, loader
