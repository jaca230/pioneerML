from __future__ import annotations

from pioneerml.common.loader.graph.event.event_splitter_graph_loader import EventSplitterGraphLoader

from .base_graph_loader_factory import BaseGraphLoaderFactory


class EventSplitterGraphLoaderFactory(BaseGraphLoaderFactory):
    def __init__(
        self,
        *,
        parquet_paths: list[str],
        group_probs_parquet_paths: list[str] | None = None,
        group_splitter_parquet_paths: list[str] | None = None,
        endpoint_parquet_paths: list[str] | None = None,
    ) -> None:
        super().__init__(parquet_paths=parquet_paths)
        self.group_probs_parquet_paths = [str(p) for p in group_probs_parquet_paths] if group_probs_parquet_paths else None
        self.group_splitter_parquet_paths = (
            [str(p) for p in group_splitter_parquet_paths] if group_splitter_parquet_paths else None
        )
        self.endpoint_parquet_paths = [str(p) for p in endpoint_parquet_paths] if endpoint_parquet_paths else None

    @staticmethod
    def _as_optional(value) -> str | None:
        if value in (None, "", "none", "None"):
            return None
        return str(value).strip().lower()

    def build_loader(self, *, loader_params: dict) -> EventSplitterGraphLoader:
        cfg = dict(loader_params or {})
        return EventSplitterGraphLoader(
            parquet_paths=list(self.parquet_paths),
            group_probs_parquet_paths=(
                list(self.group_probs_parquet_paths) if self.group_probs_parquet_paths is not None else None
            ),
            group_splitter_parquet_paths=(
                list(self.group_splitter_parquet_paths)
                if self.group_splitter_parquet_paths is not None
                else None
            ),
            endpoint_parquet_paths=(
                list(self.endpoint_parquet_paths)
                if self.endpoint_parquet_paths is not None
                else None
            ),
            mode=str(cfg.get("mode", "train")),
            use_group_probs=bool(cfg.get("use_group_probs", True)),
            use_splitter_probs=bool(cfg.get("use_splitter_probs", True)),
            use_endpoint_preds=bool(cfg.get("use_endpoint_preds", True)),
            batch_size=max(1, int(cfg.get("batch_size", 8))),
            row_groups_per_chunk=max(
                1, int(cfg.get("chunk_row_groups", cfg.get("row_groups_per_chunk", 4)))
            ),
            num_workers=max(0, int(cfg.get("chunk_workers", cfg.get("num_workers", 0)))),
            split=self._as_optional(cfg.get("split")),
            train_fraction=float(cfg.get("train_fraction", 0.9)),
            val_fraction=float(cfg.get("val_fraction", 0.05)),
            test_fraction=float(cfg.get("test_fraction", 0.05)),
            split_seed=int(cfg.get("split_seed", 0)),
            sample_fraction=(
                None
                if cfg.get("sample_fraction") in (None, "", "none", "None")
                else float(cfg.get("sample_fraction"))
            ),
        )
