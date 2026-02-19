from __future__ import annotations

from pioneerml.common.loader import GroupClassifierGraphLoaderFactory
from pioneerml.common.pipeline.services import BasePipelineService
from pioneerml.pipelines.training.group_classification.dataset import GroupClassifierDataset
from pioneerml.pipelines.training.group_classification.objective import GroupClassifierObjectiveAdapter


class GroupClassifierServiceBase(BasePipelineService):
    def __init__(
        self,
        *,
        dataset: GroupClassifierDataset,
        pipeline_config: dict | None = None,
    ) -> None:
        super().__init__(pipeline_config=pipeline_config)
        self.dataset = dataset
        factory = getattr(dataset, "loader_factory", None) or getattr(dataset, "loader", None)
        if not isinstance(factory, GroupClassifierGraphLoaderFactory):
            raise RuntimeError("Dataset is missing GroupClassifierGraphLoaderFactory.")
        self.loader_factory = factory
        self.objective_adapter = GroupClassifierObjectiveAdapter()

    def _resolve_loader_params(self, cfg: dict, *, purpose: str, forced_batch_size: int | None = None) -> dict:
        raw = cfg.get("loader_config")
        base_cfg: dict = {}
        purpose_cfg: dict = {}
        if isinstance(raw, dict):
            if any(k in raw for k in ("base", "train", "val", "evaluate", "export")):
                base_cfg = dict(raw.get("base") or {})
                purpose_cfg = dict(raw.get(purpose) or {})
            else:
                base_cfg = dict(raw)
        merged = {**base_cfg, **purpose_cfg}

        if forced_batch_size is not None:
            merged["batch_size"] = int(forced_batch_size)
        else:
            merged.setdefault("batch_size", int(cfg.get("batch_size", 64)))
        merged.setdefault("chunk_row_groups", int(cfg.get("chunk_row_groups", 4)))
        merged.setdefault("chunk_workers", int(cfg.get("chunk_workers", 0)))
        merged.setdefault("mode", "train")
        return merged

    def _merge(self, base: dict, override: dict | None) -> dict:
        return self.merge_config(base, override)
