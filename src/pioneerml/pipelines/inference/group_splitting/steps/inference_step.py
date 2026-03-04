import torch
from zenml import step

from pioneerml.common.data_loader import GroupSplitterGraphLoader
from pioneerml.common.data_writer import GroupSplittingDataWriter, TimeGroupPredictionSet
from pioneerml.common.pipeline.steps import BaseInferenceStep, InferenceSourceContext


class GroupSplitterInferenceRunStep(BaseInferenceStep):
    step_key = "inference"

    def default_config(self) -> dict:
        return {
            "threshold": 0.5,
            "materialize_outputs": None,
        }

    def expected_writer_type(self) -> type[GroupSplittingDataWriter]:
        return GroupSplittingDataWriter

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
        _ = context
        src_path = source_ctx.src_path
        n_rows = int(source_ctx.num_rows)
        group_probs_for_source = self.resolve_optional_source_paths(
            inputs=inputs,
            source_index=source_ctx.source_idx,
            total_files=len(runtime.validated_files),
            source_name="group_probs",
            input_keys=("group_probs_sources", "validated_group_probs_sources"),
        )
        loader = GroupSplitterGraphLoader(
            input_sources=self.build_input_source_for_path(
                src_path=src_path,
                optional_sources={"group_probs": group_probs_for_source},
            ),
            mode=str(inputs.get("mode", "inference")),
            data_flow_config=runtime.flow_cfg,
        ).make_dataloader(shuffle_batches=False)

        device = runtime.device
        scripted = runtime.scripted
        for batch in loader:
            x = batch.x_node.to(device, non_blocking=(device.type == "cuda"))
            edge_index = batch.edge_index.to(device, non_blocking=(device.type == "cuda"))
            edge_attr = batch.x_edge.to(device, non_blocking=(device.type == "cuda"))
            b = batch.node_graph_id.to(device, non_blocking=(device.type == "cuda"))
            group_probs = batch.x_graph.to(device, non_blocking=(device.type == "cuda"))
            logits = scripted(x, edge_index, edge_attr, b, group_probs)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            source_node_probs_np = torch.sigmoid(logits).detach().cpu().to(torch.float32).numpy().astype("float32", copy=False)
            source_node_event_ids_np = (
                batch.graph_event_id[batch.node_graph_id].to(torch.int64).cpu().numpy().astype("int64", copy=False)
            )
            source_graph_event_id_np = batch.graph_event_id.to(torch.int64).cpu().numpy().astype("int64", copy=False)
            source_graph_time_group_id_np = (
                batch.graph_time_group_id.to(torch.int64).cpu().numpy().astype("int64", copy=False)
            )
            yield TimeGroupPredictionSet(
                src_path=src_path,
                prediction_event_ids_np=source_node_event_ids_np,
                model_outputs_by_name={"main": source_node_probs_np},
                num_rows=n_rows,
                time_group_event_ids_np=source_graph_event_id_np,
                time_group_ids_np=source_graph_time_group_id_np,
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
        _ = runtime
        _ = inputs
        _ = context
        out = dict(outputs)
        out["threshold"] = float(cfg.get("threshold", 0.5))
        return out


@step(name="run_group_splitter_inference", enable_cache=False)
def run_group_splitter_inference_step(
    model_info: dict,
    inputs: dict,
    writer_setup: dict,
    pipeline_config: dict | None = None,
) -> dict:
    return GroupSplitterInferenceRunStep(pipeline_config=pipeline_config).execute(
        model_info=model_info,
        inputs=inputs,
        writer_setup=writer_setup,
    )
