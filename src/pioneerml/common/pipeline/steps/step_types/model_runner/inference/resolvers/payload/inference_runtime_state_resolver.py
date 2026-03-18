from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from pioneerml.common.data_loader.factory import LoaderFactory
from pioneerml.common.data_loader.input_source import InputSourceSet
from pioneerml.common.data_loader.input_source import create_input_backend
from pioneerml.common.data_writer import WriterFactory
from pioneerml.common.integration.pytorch.model_handle import BaseModelHandle

from ......resolver import BasePayloadResolver


class InferenceRuntimeStateResolver(BasePayloadResolver):
    def resolve(
        self,
        *,
        payloads: Mapping[str, Any] | None,
        runtime_state: dict[str, Any],
    ) -> None:
        if not isinstance(payloads, Mapping):
            raise RuntimeError(
                f"{self.step.__class__.__name__} requires payloads containing "
                "writer_factory_init and model_handle_builder outputs."
            )

        writer_payload = payloads.get("writer_factory_init")
        model_payload = payloads.get("model_handle_builder")

        if not isinstance(writer_payload, Mapping):
            raise RuntimeError("Inference payloads missing mapping key 'writer_factory_init'.")
        if not isinstance(model_payload, Mapping):
            raise RuntimeError("Inference payloads missing mapping key 'model_handle_builder'.")

        writer_factory = writer_payload.get("writer_factory")
        model_handle = model_payload.get("model_handle")
        loader_factory = runtime_state.get("loader_factory")

        if not isinstance(loader_factory, LoaderFactory):
            raise RuntimeError(
                f"{self.step.__class__.__name__} runtime_state missing valid 'loader_factory'. "
                "This should be resolved by ModelRunnerPayloadResolver."
            )
        if not isinstance(writer_factory, WriterFactory):
            raise RuntimeError("Inference payloads missing valid 'writer_factory_init.writer_factory'.")
        if not isinstance(model_handle, BaseModelHandle):
            raise RuntimeError("Inference payloads missing valid 'model_handle_builder.model_handle'.")

        runtime_state["inference_runtime"] = self._build_runtime(
            cfg=dict(self.step.config_json),
            loader_factory=loader_factory,
            writer_factory=writer_factory,
            model_handle=model_handle,
        )

    def _build_runtime(
        self,
        *,
        cfg: dict[str, Any],
        loader_factory: LoaderFactory,
        writer_factory: WriterFactory,
        model_handle: BaseModelHandle,
    ) -> dict[str, Any]:
        writer = writer_factory.create()

        use_cuda = bool(cfg.get("use_cuda", True))
        device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        model = model_handle.load(device=device)

        loader_params = LoaderFactory._resolve_loader_params(cfg, purpose="inference", default_mode="inference")
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
            "output_path": (None if writer_factory.output_path is None else str(writer_factory.output_path)),
        }

    @staticmethod
    def _collect_validated_files(*, loader_factory: LoaderFactory) -> list[str]:
        files = [str(path) for path in list(loader_factory.input_sources.main_sources)]
        if not files:
            raise RuntimeError("No validated files provided for inference.")
        return files

    @staticmethod
    def _collect_validated_file_rows(
        *,
        loader_factory: LoaderFactory,
        validated_files: list[str],
    ) -> list[int]:
        backend = create_input_backend(str(loader_factory.input_backend_name))
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
        source_input_sources = self._build_input_sources_for_source(
            input_sources=loader_factory.input_sources,
            source_index=int(source_index),
        )
        source_loader_factory = LoaderFactory(
            loader_cls=loader_factory.loader_cls,
            loader_name=loader_factory.loader_name,
            input_sources=source_input_sources,
            input_backend_name=loader_factory.input_backend_name,
            default_mode=loader_factory.default_mode,
        )
        return source_loader_factory.build_loader(loader_params=dict(loader_params))
