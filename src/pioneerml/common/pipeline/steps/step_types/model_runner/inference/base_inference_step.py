from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from typing import Any

import torch

from pioneerml.common.data_writer.input_source import PredictionSet

from .payloads import InferenceStepPayload
from .resolvers import InferenceConfigResolver, InferenceStateResolver
from ..base_model_runner_step import BaseModelRunnerStep
from ..utils import merge_nested_dicts


class BaseInferenceStep(BaseModelRunnerStep):
    prediction_set_cls: type[PredictionSet] = PredictionSet
    DEFAULT_CONFIG = merge_nested_dicts(
        base=BaseModelRunnerStep.DEFAULT_CONFIG,
        override={
            "threshold": 0.5,
            "use_cuda": True,
            "loader_manager": {
                "config": {
                    "defaults": {
                        "type": "group_classifier",
                        "config": {
                            "mode": "inference",
                            "batch_size": 64,
                            "chunk_row_groups": 4,
                            "chunk_workers": 0,
                            "sample_fraction": 1.0,
                            "split_seed": 0,
                        },
                    },
                    "loaders": {
                        "inference_loader": {
                            "config": {
                                "mode": "inference",
                                "shuffle_batches": False,
                                "log_diagnostics": False,
                            },
                        },
                    },
                },
            },
        },
    )
    config_resolver_classes = BaseModelRunnerStep.config_resolver_classes + (InferenceConfigResolver,)
    payload_resolver_classes = BaseModelRunnerStep.payload_resolver_classes + (InferenceStateResolver,)

    @abstractmethod
    def build_model_input(
        self,
        *,
        batch,
        device: torch.device,
        cfg: dict,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        raise NotImplementedError(f"{self.__class__.__name__} must implement build_model_input(...).")

    @abstractmethod
    def build_prediction_fragment(
        self,
        *,
        batch,
        model_output,
        cfg: dict,
    ) -> Mapping[str, Any] | PredictionSet:
        raise NotImplementedError(f"{self.__class__.__name__} must implement build_prediction_fragment(...).")

    def _execute(self, *, inputs: dict | None = None) -> InferenceStepPayload:
        _ = inputs
        cfg = dict(self.config_json)

        runtime = self.runtime_state.get("inference_runtime")
        if not isinstance(runtime, dict):
            raise RuntimeError("Inference runtime_state missing valid 'inference_runtime'.")

        writer = runtime.get("writer")
        source_items = runtime.get("source_items")
        if writer is None or not hasattr(writer, "on_start"):
            raise RuntimeError("Inference runtime missing valid writer object.")
        if not isinstance(source_items, list):
            raise RuntimeError("Inference runtime missing valid source_items list.")

        writer_cfg = writer.run_config
        chunk_output_path = runtime.get("output_path") if len(source_items) == 1 else None

        start_state = {
            "source_contexts": [{"src_path": str(i["src_path"]), "num_rows": int(i["num_rows"])} for i in source_items],
            "output_dir": writer_cfg.output_dir,
            "output_path": runtime.get("output_path"),
            "write_timestamped": bool(writer_cfg.write_timestamped),
            "timestamp": writer_cfg.timestamp,
            "streaming": bool(writer_cfg.streaming),
        }
        writer.on_start(state=start_state)

        shuffle_batches = bool(runtime.get("shuffle_batches", False))
        model = runtime.get("model")
        device = runtime.get("device")
        with torch.no_grad():
            for source_item in source_items:
                source_loader = source_item.get("loader")
                if source_loader is None or not hasattr(source_loader, "make_dataloader"):
                    raise RuntimeError("Inference runtime source item missing valid loader provider.")

                for batch in source_loader.make_dataloader(shuffle_batches=shuffle_batches):
                    model_args, model_kwargs = self.build_model_input(
                        batch=batch,
                        device=device,
                        cfg=cfg,
                    )
                    if not isinstance(model_args, tuple):
                        raise RuntimeError(
                            f"{self.__class__.__name__}.build_model_input(...) must return tuple args as first element."
                        )
                    if not isinstance(model_kwargs, dict):
                        raise RuntimeError(
                            f"{self.__class__.__name__}.build_model_input(...) must return dict kwargs as second element."
                        )

                    model_output = model(*model_args, **model_kwargs)
                    fragment = self.build_prediction_fragment(
                        batch=batch,
                        model_output=model_output,
                        cfg=cfg,
                    )
                    if isinstance(fragment, PredictionSet):
                        prediction_set = fragment
                        writer.on_chunk(
                            state=writer.chunk_state(
                                prediction_set=prediction_set,
                                output_dir=writer_cfg.output_dir,
                                output_path=chunk_output_path,
                                write_timestamped=bool(writer_cfg.write_timestamped),
                                timestamp=writer_cfg.timestamp,
                            )
                        )
                        continue

                    if not isinstance(fragment, Mapping):
                        raise RuntimeError(
                            f"{self.__class__.__name__}.build_prediction_fragment(...) must return dict fragment "
                            "or PredictionSet."
                        )
                    pred_cls = getattr(self, "prediction_set_cls", None)
                    if not isinstance(pred_cls, type) or not issubclass(pred_cls, PredictionSet):
                        raise RuntimeError(
                            f"{self.__class__.__name__}.prediction_set_cls must be a PredictionSet subclass."
                        )

                    payload = dict(fragment)
                    payload.pop("src_path", None)
                    payload.pop("num_rows", None)
                    prediction_set = pred_cls(
                        src_path=source_item["src_path"],
                        num_rows=int(source_item["num_rows"]),
                        **payload,
                    )
                    writer.on_chunk(
                        state=writer.chunk_state(
                            prediction_set=prediction_set,
                            output_dir=writer_cfg.output_dir,
                            output_path=chunk_output_path,
                            write_timestamped=bool(writer_cfg.write_timestamped),
                            timestamp=writer_cfg.timestamp,
                        )
                    )

        finalized = writer.on_finalize(state=start_state)
        run_outputs = dict(finalized.get("run_outputs") or {})
        prediction_paths = [str(path) for path in run_outputs.get("predictions_paths") or []]
        timestamped_prediction_paths = [str(path) for path in run_outputs.get("timestamped_predictions_paths") or []]

        return InferenceStepPayload(
            predictions_path=prediction_paths[0] if len(prediction_paths) == 1 else None,
            predictions_paths=prediction_paths,
            timestamped_predictions_path=(
                timestamped_prediction_paths[0] if len(timestamped_prediction_paths) == 1 else None
            ),
            timestamped_predictions_paths=timestamped_prediction_paths,
        )
