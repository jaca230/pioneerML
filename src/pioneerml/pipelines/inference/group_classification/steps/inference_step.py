
import torch
from zenml import step

from pioneerml.common.data_loader import GroupClassifierGraphLoader
from pioneerml.common.data_writer import GroupClassificationDataWriter, TimeGroupPredictionSet
from pioneerml.common.pipeline.steps import BaseInferenceStep, InferenceSourceContext


class GroupClassifierInferenceRunStep(BaseInferenceStep):
    step_key = "inference"

    def default_config(self) -> dict:
        return {
            "threshold": 0.5,
            "materialize_outputs": None,
        }

    def expected_writer_type(self) -> type[GroupClassificationDataWriter]:
        return GroupClassificationDataWriter

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
        loader = GroupClassifierGraphLoader(
            input_sources=self.build_input_source_for_path(src_path=src_path),
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
            logits = scripted(x, edge_index, edge_attr, b)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            probs_np = torch.sigmoid(logits).detach().cpu().to(torch.float32).numpy().astype("float32", copy=False)
            graph_event_ids_np = batch.graph_event_id.to(torch.int64).cpu().numpy().astype("int64", copy=False)
            graph_time_group_ids_np = batch.graph_time_group_id.to(torch.int64).cpu().numpy().astype("int64", copy=False)
            yield TimeGroupPredictionSet(
                src_path=src_path,
                prediction_event_ids_np=graph_event_ids_np,
                model_outputs_by_name={"main": probs_np},
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
        _ = runtime
        _ = inputs
        _ = context
        out = dict(outputs)
        out["threshold"] = float(cfg.get("threshold", 0.5))
        return out


@step(name="run_group_classifier_inference", enable_cache=False)
def run_group_classifier_inference_step(
    model_info: dict,
    inputs: dict,
    writer_setup: dict,
    pipeline_config: dict | None = None,
) -> dict:
    return GroupClassifierInferenceRunStep(pipeline_config=pipeline_config).execute(
        model_info=model_info,
        inputs=inputs,
        writer_setup=writer_setup,
    )
