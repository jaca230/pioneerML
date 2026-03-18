from __future__ import annotations

from .payloads import ExportStepPayload
from .resolvers import ExportRuntimeConfigResolver, ExportRuntimeStateResolver
from .utils.export_utils import (
    build_data_shapes,
    build_export_paths,
    export_model_artifact,
    json_safe,
    write_export_metadata,
)
from ..base_model_runner_step import BaseModelRunnerStep
from ..utils import merge_nested_dicts


class BaseExportStep(BaseModelRunnerStep):
    DEFAULT_CONFIG = merge_nested_dicts(
        base=BaseModelRunnerStep.DEFAULT_CONFIG,
        override={
            "enabled": True,
            "export_type": "script",
            "export_dir": None,
            "filename_prefix": None,
            "prefer_cuda": True,
            "loader_config": {
                "base": {
                    "batch_size": 1,
                    "chunk_row_groups": 1,
                    "chunk_workers": 0,
                },
                "export": {
                    "mode": "train",
                    "shuffle_batches": False,
                    "log_diagnostics": False,
                },
            },
        },
    )
    config_resolver_classes = BaseModelRunnerStep.config_resolver_classes + (ExportRuntimeConfigResolver,)
    payload_resolver_classes = BaseModelRunnerStep.payload_resolver_classes + (ExportRuntimeStateResolver,)

    def _build_payload(
        self,
        *,
        torchscript_path: str | None,
        metadata_path: str | None,
        **kwargs,
    ) -> ExportStepPayload:
        payload = {
            "torchscript_path": torchscript_path,
            "metadata_path": metadata_path,
        }
        payload.update(dict(kwargs))
        return ExportStepPayload(**payload)

    def _execute(self) -> ExportStepPayload:
        cfg = dict(self.config_json)
        if not bool(cfg["enabled"]):
            return self._build_payload(torchscript_path=None, metadata_path=None, skipped=True)

        module = self.runtime_state.get("module")
        dataset = self.runtime_state.get("export_dataset")
        export_provider = self.runtime_state.get("export_provider")
        if module is None:
            raise RuntimeError(f"{self.__class__.__name__} runtime_state missing 'module'.")
        if dataset is None:
            raise RuntimeError(f"{self.__class__.__name__} runtime_state missing 'export_dataset'.")
        if export_provider is None:
            raise RuntimeError(f"{self.__class__.__name__} runtime_state missing 'export_provider'.")

        model_obj = getattr(module, "model", module)
        if model_obj is None:
            return self._build_payload(torchscript_path=None, metadata_path=None, skipped=True)

        export_dir = str(cfg["export_dir"])
        filename_prefix = str(cfg["filename_prefix"])
        export_type = str(cfg["export_type"])
        timestamp, torchscript_path, metadata_path = build_export_paths(
            export_dir=export_dir,
            filename_prefix=filename_prefix,
            export_type=export_type,
        )

        export_model_artifact(
            model_obj=model_obj,
            output_path=torchscript_path,
            export_type=export_type,
            prefer_cuda=bool(cfg.get("prefer_cuda", True)),
            cfg=cfg,
            dataset=dataset,
            loader_provider=export_provider,
        )

        data_shapes = build_data_shapes(dataset=dataset)
        meta = {
            "timestamp": timestamp,
            "torchscript_path": str(torchscript_path),
            "pipeline_config": json_safe(self.pipeline_config or {}),
            "export_config": json_safe(cfg),
            "hpo_params": json_safe(self.runtime_state.get("hpo_params") or {}),
            "metrics": json_safe(self.runtime_state.get("metrics") or {}),
            "export_type": export_type,
            "data_shapes": data_shapes,
        }
        write_export_metadata(metadata_path=metadata_path, metadata=meta)

        return self._build_payload(
            torchscript_path=str(torchscript_path),
            metadata_path=str(metadata_path),
            export_type=export_type,
        )
