from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from pioneerml.data_loader import BaseLoaderManager, LoaderFactory
from pioneerml.data_loader.loaders.input_source import InputBackend, InputSourceSet
from pioneerml.data_writer import WriterFactory
from pioneerml.integration.pytorch.model_handles import BaseModelHandle

from ......resolver import BasePayloadResolver


class InferenceStateResolver(BasePayloadResolver):
    def resolve(
        self,
        *,
        payloads: Mapping[str, Any] | None,
        runtime_state: dict[str, Any],
    ) -> None:
        if not isinstance(payloads, Mapping):
            raise RuntimeError(
                f"{self.step.__class__.__name__} requires payloads containing model_handle_builder output."
            )

        model_payload = payloads.get("model_handle_builder")

        if not isinstance(model_payload, Mapping):
            raise RuntimeError("Inference payloads missing mapping key 'model_handle_builder'.")

        writer_factory = runtime_state.get("writer_factory")
        model_handle = model_payload.get("model_handle")
        loader_manager = runtime_state.get("loader_manager")

        if not isinstance(loader_manager, BaseLoaderManager):
            raise RuntimeError(
                f"{self.step.__class__.__name__} runtime_state missing valid 'loader_manager'. "
                "This should be resolved by ModelRunnerStateResolver."
            )
        if not isinstance(writer_factory, WriterFactory):
            raise RuntimeError("Inference runtime_state missing valid 'writer_factory'.")
        if not isinstance(model_handle, BaseModelHandle):
            raise RuntimeError("Inference payloads missing valid 'model_handle_builder.model_handle'.")

        runtime_state["inference_runtime"] = self._build_runtime(
            cfg=dict(self.step.config_json),
            loader_manager=loader_manager,
            writer_factory=writer_factory,
            model_handle=model_handle,
        )

    def _build_runtime(
        self,
        *,
        cfg: dict[str, Any],
        loader_manager: BaseLoaderManager,
        writer_factory: WriterFactory,
        model_handle: BaseModelHandle,
    ) -> dict[str, Any]:
        writer = writer_factory.build()
        if not hasattr(writer, "build_prediction_set"):
            raise RuntimeError(
                f"{writer.__class__.__name__} must implement build_prediction_set(...) for inference usage."
            )
        loader_factory = loader_manager.loader_factory

        runtime_cfg = dict(cfg.get("runtime") or {})
        prefer_cuda = bool(runtime_cfg.get("prefer_cuda", True))
        device = torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")
        model = model_handle.load(device=device)

        loader_params = loader_manager.resolve_loader_params(
            purpose="inference",
            default_mode="inference",
        )
        validated_files = self._collect_validated_files(loader_factory=loader_factory)
        validated_file_rows = self._collect_validated_file_rows(
            loader_factory=loader_factory,
            validated_files=validated_files,
        )
        source_contexts = self._build_source_contexts(
            validated_files=validated_files,
            validated_file_rows=validated_file_rows,
        )

        source_items: list[dict[str, Any]] = []
        for source_ctx in source_contexts:
            source_loader = self._build_source_loader_provider(
                loader_factory=loader_factory,
                source_index=int(source_ctx["source_idx"]),
                loader_params=loader_params,
            )
            if not hasattr(source_loader, "build_inference_model_input"):
                raise RuntimeError(
                    f"{source_loader.__class__.__name__} must implement build_inference_model_input(...) for inference usage."
                )
            source_items.append(
                {
                    "source_idx": int(source_ctx["source_idx"]),
                    "src_path": source_ctx["src_path"],
                    "num_rows": int(source_ctx["num_rows"]),
                    "loader": source_loader,
                }
            )

        return {
            "writer": writer,
            "device": device,
            "model": model,
            "source_items": source_items,
            "shuffle_batches": bool(loader_params.get("shuffle_batches", False)),
            "shuffle_within_batch": bool(loader_params.get("shuffle_within_batch", False)),
            "output_path": (None if writer_factory.config.get("output_path") is None else str(writer_factory.config.get("output_path"))),
        }

    @staticmethod
    def _collect_validated_files(*, loader_factory: LoaderFactory) -> list[str]:
        input_sources = loader_factory.config.get("input_sources")
        if not isinstance(input_sources, InputSourceSet):
            raise RuntimeError("LoaderFactory config missing valid 'input_sources'.")
        files = [str(path) for path in list(input_sources.main_sources)]
        if not files:
            raise RuntimeError("No validated files provided for inference.")
        return files

    @staticmethod
    def _collect_validated_file_rows(
        *,
        loader_factory: LoaderFactory,
        validated_files: list[str],
    ) -> list[int]:
        backend = loader_factory.config.get("input_backend")
        if not isinstance(backend, InputBackend):
            raise RuntimeError("LoaderFactory config missing valid 'input_backend'.")
        counted = backend.count_rows_per_source(sources=tuple(str(path) for path in validated_files))
        rows = [int(v) for v in counted]

        if len(rows) != len(validated_files):
            raise RuntimeError(
                "Expected validated_file_rows aligned with validated_files "
                f"({len(validated_files)}), got {len(rows)}."
            )
        return rows

    @staticmethod
    def _build_source_contexts(
        *,
        validated_files: list[str],
        validated_file_rows: list[int],
    ) -> list[dict[str, Any]]:
        if len(validated_file_rows) != len(validated_files):
            raise RuntimeError(
                "validated_file_rows must align with validated_files. "
                f"Got {len(validated_file_rows)} vs {len(validated_files)}."
            )

        out: list[dict[str, Any]] = []
        source_event_offset = 0
        for source_idx, (src_file, num_rows) in enumerate(zip(validated_files, validated_file_rows, strict=True)):
            out.append(
                {
                    "source_idx": int(source_idx),
                    "src_path": Path(src_file).expanduser().resolve(),
                    "num_rows": int(num_rows),
                    "source_event_offset": int(source_event_offset),
                }
            )
            source_event_offset += int(num_rows)
        return out

    @staticmethod
    def _build_input_sources_for_source(
        *,
        input_sources: InputSourceSet,
        source_index: int,
    ) -> InputSourceSet:
        main_source = str(input_sources.main_sources[source_index])
        dynamic_sources: dict[str, list[str] | None] = {}
        for name, aligned_sources in input_sources.optional_sources_by_name.items():
            values = list(aligned_sources)
            dynamic_sources[str(name)] = [str(values[source_index])] if len(values) > source_index else None
        return InputSourceSet(
            main_sources=[main_source],
            optional_sources_by_name=dynamic_sources,
            source_type=input_sources.source_type,
        )

    def _build_source_loader_provider(
        self,
        *,
        loader_factory: LoaderFactory,
        source_index: int,
        loader_params: Mapping[str, Any],
    ):
        input_sources = loader_factory.config.get("input_sources")
        if not isinstance(input_sources, InputSourceSet):
            raise RuntimeError("LoaderFactory config missing valid 'input_sources'.")
        source_input_sources = self._build_input_sources_for_source(
            input_sources=input_sources,
            source_index=int(source_index),
        )
        source_loader_factory = LoaderFactory(
            loader_cls=loader_factory.plugin_cls,
            loader_name=loader_factory.plugin_name,
            config={**dict(loader_factory.config), "input_sources": source_input_sources},
        )
        return source_loader_factory.build(
            config=dict(loader_params)
        )
