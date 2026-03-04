import torch
from zenml import step

from pioneerml.common.data_loader import EndpointRegressionGraphLoader
from pioneerml.common.data_writer import EndpointRegressionDataWriter, TimeGroupPredictionSet
from pioneerml.common.pipeline.steps import BaseInferenceStep, InferenceSourceContext


class EndpointRegressorInferenceRunStep(BaseInferenceStep):
    step_key = "inference"

    def default_config(self) -> dict:
        return {
            "materialize_outputs": None,
        }

    @staticmethod
    def _script_inputs(batch, device: torch.device):
        x = batch.x_node.to(device, non_blocking=(device.type == "cuda"))
        edge_index = batch.edge_index.to(device, non_blocking=(device.type == "cuda"))
        edge_attr = batch.x_edge.to(device, non_blocking=(device.type == "cuda"))
        node_graph_id = batch.node_graph_id.to(device, non_blocking=(device.type == "cuda"))
        x_graph = batch.x_graph.to(device, non_blocking=(device.type == "cuda"))
        group_probs = x_graph[:, :3]
        u = x_graph[:, 3:4] if x_graph.shape[1] >= 4 else x.new_zeros((x_graph.shape[0], 1))
        splitter_probs = x.new_zeros((x.shape[0], 0))
        return x, edge_index, edge_attr, node_graph_id, u, group_probs, splitter_probs

    def expected_writer_type(self) -> type[EndpointRegressionDataWriter]:
        return EndpointRegressionDataWriter

    def prepare_inference_context(
        self,
        *,
        cfg: dict,
        runtime,
        inputs: dict,
    ) -> dict:
        _ = cfg
        _ = runtime
        _ = inputs
        return {"prediction_dim": None}

    def infer_prediction_sets_for_source(
        self,
        *,
        source_ctx: InferenceSourceContext,
        runtime,
        cfg: dict,
        inputs: dict,
        context: dict,
    ):
        _ = cfg
        src_path = source_ctx.src_path
        n_rows = int(source_ctx.num_rows)
        group_probs_for_source = self.resolve_optional_source_paths(
            inputs=inputs,
            source_index=source_ctx.source_idx,
            total_files=len(runtime.validated_files),
            source_name="group_probs",
            input_keys=("validated_group_probs_sources", "group_probs_sources"),
        )
        splitter_for_source = self.resolve_optional_source_paths(
            inputs=inputs,
            source_index=source_ctx.source_idx,
            total_files=len(runtime.validated_files),
            source_name="group_splitter",
            input_keys=("validated_group_splitter_sources", "group_splitter_sources"),
        )

        loader = EndpointRegressionGraphLoader(
            input_sources=self.build_input_source_for_path(
                src_path=src_path,
                optional_sources={
                    "group_probs": group_probs_for_source,
                    "group_splitter": splitter_for_source,
                },
            ),
            mode=str(inputs.get("mode", "inference")),
            data_flow_config=runtime.flow_cfg,
        ).make_dataloader(shuffle_batches=False)

        device = runtime.device
        scripted = runtime.scripted
        for batch in loader:
            script_inputs = self._script_inputs(batch, device)
            preds = scripted(*script_inputs)
            preds = preds[0] if isinstance(preds, (tuple, list)) else preds
            preds_cpu = preds.detach().cpu().to(torch.float32)
            context["prediction_dim"] = int(preds_cpu.shape[1]) if preds_cpu.ndim == 2 else context.get("prediction_dim")
            source_preds_np = preds_cpu.numpy().astype("float32", copy=False)
            graph_event_ids_np = batch.graph_event_id.to(torch.int64).cpu().numpy().astype("int64", copy=False)
            graph_time_group_ids_np = batch.graph_time_group_id.to(torch.int64).cpu().numpy().astype("int64", copy=False)
            yield TimeGroupPredictionSet(
                src_path=src_path,
                prediction_event_ids_np=graph_event_ids_np,
                model_outputs_by_name={"main": source_preds_np},
                num_rows=n_rows,
                time_group_event_ids_np=graph_event_ids_np,
                time_group_ids_np=graph_time_group_ids_np,
            )

    def finalize_inference_outputs(
        self,
        *,
        outputs: dict,
        cfg: dict,
        runtime,
        inputs: dict,
        context: dict,
    ) -> dict:
        _ = cfg
        _ = runtime
        out = dict(outputs)
        out["prediction_dim"] = int(context.get("prediction_dim")) if context.get("prediction_dim") is not None else None
        out["validated_group_probs_sources"] = [str(p) for p in inputs.get("validated_group_probs_sources") or []]
        out["validated_group_splitter_sources"] = [str(p) for p in inputs.get("validated_group_splitter_sources") or []]
        return out


@step(name="run_endpoint_regressor_inference", enable_cache=False)
def run_endpoint_regressor_inference_step(
    model_info: dict,
    inputs: dict,
    writer_setup: dict,
    pipeline_config: dict | None = None,
) -> dict:
    return EndpointRegressorInferenceRunStep(pipeline_config=pipeline_config).execute(
        model_info=model_info,
        inputs=inputs,
        writer_setup=writer_setup,
    )
