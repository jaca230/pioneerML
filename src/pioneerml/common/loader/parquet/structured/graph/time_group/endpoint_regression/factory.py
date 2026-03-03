from __future__ import annotations

from pioneerml.common.loader.config import DataFlowConfig, SplitSampleConfig
from pioneerml.common.parquet import ParquetInputSet

from .loader import EndpointRegressionGraphLoader


class EndpointRegressionGraphLoaderFactory:
    def __init__(
        self,
        *,
        parquet_paths: list[str] | None = None,
        parquet_inputs: ParquetInputSet | None = None,
        group_probs_parquet_paths: list[str] | None = None,
        group_splitter_parquet_paths: list[str] | None = None,
    ) -> None:
        self.parquet_inputs = (
            parquet_inputs
            if parquet_inputs is not None
            else ParquetInputSet(
                main_paths=[str(p) for p in (parquet_paths or [])],
                optional_paths_by_name={
                    "group_probs": [str(p) for p in group_probs_parquet_paths] if group_probs_parquet_paths else None,
                    "group_splitter": (
                        [str(p) for p in group_splitter_parquet_paths] if group_splitter_parquet_paths else None
                    ),
                },
            )
        )

    @staticmethod
    def _as_optional(value) -> str | None:
        if value in (None, "", "none", "None"):
            return None
        return str(value).strip().lower()

    def build_loader(self, *, loader_params: dict) -> EndpointRegressionGraphLoader:
        cfg = dict(loader_params or {})
        split_seed_raw = cfg.get("split_seed", None)
        split_seed = None if split_seed_raw in (None, "", "none", "None") else int(split_seed_raw)
        split_cfg = SplitSampleConfig(
            split=self._as_optional(cfg.get("split")),
            train_fraction=float(cfg.get("train_fraction", 0.9)),
            val_fraction=float(cfg.get("val_fraction", 0.05)),
            test_fraction=float(cfg.get("test_fraction", 0.05)),
            split_seed=split_seed,
            sample_fraction=(
                None if cfg.get("sample_fraction") in (None, "", "none", "None") else float(cfg.get("sample_fraction"))
            ),
        )
        data_flow_cfg = DataFlowConfig(
            batch_size=max(1, int(cfg.get("batch_size", 64))),
            row_groups_per_chunk=max(1, int(cfg.get("chunk_row_groups", cfg.get("row_groups_per_chunk", 4)))),
            num_workers=max(0, int(cfg.get("chunk_workers", cfg.get("num_workers", 0)))),
        )
        return EndpointRegressionGraphLoader(
            parquet_inputs=self.parquet_inputs,
            mode=str(cfg.get("mode", "train")),
            data_flow_config=data_flow_cfg,
            split_config=split_cfg,
            profiling=dict(cfg.get("profiling") or {}),
        )
