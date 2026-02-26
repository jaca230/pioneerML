from __future__ import annotations

from ..graph.time_group.group_classifier_graph_loader import GroupClassifierGraphLoader

from .base_graph_loader_factory import BaseGraphLoaderFactory


class GroupClassifierGraphLoaderFactory(BaseGraphLoaderFactory):
    @staticmethod
    def _as_optional(value) -> str | None:
        if value in (None, "", "none", "None"):
            return None
        return str(value).strip().lower()

    def build_loader(self, *, loader_params: dict) -> GroupClassifierGraphLoader:
        cfg = dict(loader_params or {})
        implementation = str(cfg.get("implementation", cfg.get("loader_impl", "legacy"))).strip().lower()
        return GroupClassifierGraphLoader(
            parquet_paths=list(self.parquet_paths),
            mode=str(cfg.get("mode", "train")),
            batch_size=max(1, int(cfg.get("batch_size", 64))),
            row_groups_per_chunk=max(1, int(cfg.get("chunk_row_groups", cfg.get("row_groups_per_chunk", 4)))),
            num_workers=max(0, int(cfg.get("chunk_workers", cfg.get("num_workers", 0)))),
            split=self._as_optional(cfg.get("split")),
            train_fraction=float(cfg.get("train_fraction", 0.9)),
            val_fraction=float(cfg.get("val_fraction", 0.05)),
            test_fraction=float(cfg.get("test_fraction", 0.05)),
            split_seed=int(cfg.get("split_seed", 0)),
            sample_fraction=(
                None if cfg.get("sample_fraction") in (None, "", "none", "None") else float(cfg.get("sample_fraction"))
            ),
            implementation=implementation,
        )
