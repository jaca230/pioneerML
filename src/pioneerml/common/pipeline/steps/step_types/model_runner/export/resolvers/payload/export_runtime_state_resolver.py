from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

from pioneerml.common.data_loader import LoaderFactory

from ......resolver import BasePayloadResolver


class ExportRuntimeStateResolver(BasePayloadResolver):
    def resolve(
        self,
        *,
        payloads: Mapping[str, Any] | None,
        runtime_state: dict[str, Any],
    ) -> None:
        module = self._resolve_module(runtime_state=runtime_state)
        loader_factory = runtime_state.get("loader_factory")
        if not isinstance(loader_factory, LoaderFactory):
            raise RuntimeError(
                f"{self.step.__class__.__name__} runtime_state missing valid 'loader_factory'."
            )
        hpo_params = self._resolve_optional_dict(payloads=payloads, key="hpo_params")
        metrics = self._resolve_optional_dict(payloads=payloads, key="metrics")

        params = LoaderFactory._resolve_loader_params(
            {
                "batch_size": 1,
                "chunk_row_groups": 1,
                "chunk_workers": 0,
                "loader_config": self.step.config_json.get("loader_config"),
            },
            purpose="export",
            forced_batch_size=1,
        )
        provider = loader_factory.build_loader(loader_params=params)
        data = provider.empty_data()
        data.source_main_sources = list(provider.input_sources.main_sources)
        dataset_for_export = SimpleNamespace(data=data)

        runtime_state["module"] = module
        runtime_state["loader_factory"] = loader_factory
        runtime_state["export_provider"] = provider
        runtime_state["export_loader_params"] = dict(params)
        runtime_state["export_dataset"] = dataset_for_export
        runtime_state["hpo_params"] = hpo_params
        runtime_state["metrics"] = metrics
        runtime_state["upstream_payloads"] = dict(payloads or {})

    def _resolve_module(self, *, runtime_state: dict[str, Any]) -> Any:
        upstream_payloads = runtime_state.get("upstream_payloads")
        if isinstance(upstream_payloads, Mapping):
            train_payload = upstream_payloads.get("train") or upstream_payloads.get("train_payload")
            if isinstance(train_payload, Mapping):
                module = train_payload.get("module")
                if module is not None:
                    return module
            for value in upstream_payloads.values():
                if isinstance(value, Mapping):
                    candidate = value.get("module")
                    if candidate is not None:
                        return candidate
            direct = upstream_payloads.get("module")
            if isinstance(direct, Mapping):
                candidate = direct.get("module")
                if candidate is not None:
                    return candidate
            if direct is not None:
                return direct
        raise RuntimeError(
            f"{self.step.__class__.__name__} requires an upstream payload containing 'module'."
        )

    @staticmethod
    def _resolve_optional_dict(*, payloads: Mapping[str, Any] | None, key: str) -> dict[str, Any]:
        if isinstance(payloads, Mapping):
            direct = payloads.get(key)
            if isinstance(direct, Mapping):
                return dict(direct)
            for value in payloads.values():
                if isinstance(value, Mapping) and isinstance(value.get(key), Mapping):
                    return dict(value.get(key) or {})
        return {}
