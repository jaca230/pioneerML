from __future__ import annotations

from pioneerml.integration.pytorch.exporters import BaseExporter, ExporterFactory

from .payloads import ExportStepPayload
from .resolvers import ExportConfigResolver, ExportStateResolver
from .utils.export_utils import (
    build_data_shapes,
    json_safe,
    write_export_metadata,
)
from ..base_model_runner_step import BaseModelRunnerStep
from ..utils import merge_nested_dicts


class BaseExportStep(BaseModelRunnerStep):
    DEFAULT_CONFIG = merge_nested_dicts(
        base=BaseModelRunnerStep.DEFAULT_CONFIG,
        override={
            "exporter": {
                "type": "torchscript",
                "config": {
                    "enabled": True,
                    "export_dir": None,
                    "filename_prefix": None,
                    "prefer_cuda": True,
                },
            },
            "loader_manager": {
                "config": {
                    "defaults": {
                        "type": "group_classifier",
                        "config": {
                            "batch_size": 1,
                            "chunk_row_groups": 1,
                            "chunk_workers": 0,
                        },
                    },
                    "loaders": {
                        "export_loader": {
                            "config": {
                                "mode": "train",
                                "shuffle_batches": False,
                                "log_diagnostics": False,
                            },
                        },
                    },
                },
            },
        },
    )
    config_resolver_classes = BaseModelRunnerStep.config_resolver_classes + (ExportConfigResolver,)
    payload_resolver_classes = BaseModelRunnerStep.payload_resolver_classes + (ExportStateResolver,)

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
        exporter_cfg = dict(cfg.get("exporter") or {})
        exporter_type = str(exporter_cfg.get("type") or "").strip()
        exporter_config = dict(exporter_cfg.get("config") or {})
        if not bool(exporter_config.get("enabled", True)):
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

        export_dir = str(exporter_config["export_dir"])
        filename_prefix = str(exporter_config["filename_prefix"])
        exporter = ExporterFactory(exporter_name=exporter_type).build(config=exporter_config)
        if not isinstance(exporter, BaseExporter):
            raise RuntimeError(f"{self.__class__.__name__} exporter_factory must build BaseExporter.")
        timestamp, torchscript_path, metadata_path = exporter.build_paths(
            export_dir=export_dir,
            filename_prefix=filename_prefix,
        )

        exporter.export(
            model_obj=model_obj,
            output_path=torchscript_path,
            prefer_cuda=bool(exporter_config.get("prefer_cuda", True)),
            cfg=exporter_config,
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
            "export_type": str(exporter.export_type),
            "exporter_type": exporter_type,
            "data_shapes": data_shapes,
        }
        write_export_metadata(metadata_path=metadata_path, metadata=meta)

        return self._build_payload(
            torchscript_path=str(torchscript_path),
            metadata_path=str(metadata_path),
            export_type=str(exporter.export_type),
            exporter_type=exporter_type,
        )
