
import torch
from zenml import step

from pioneerml.common.data_writer import TimeGroupPredictionSet
from pioneerml.common.pipeline.steps import BaseInferenceStep


class GroupClassifierInferenceRunStep(BaseInferenceStep):
    step_key = "inference"
    prediction_set_cls = TimeGroupPredictionSet

    def default_config(self) -> dict:
        return {
            "threshold": 0.5,
        }

    def build_model_input(
        self,
        *,
        batch,
        device: torch.device,
        cfg: dict,
    ) -> tuple[tuple, dict]:
        _ = cfg
        x = batch.x_node.to(device, non_blocking=(device.type == "cuda"))
        edge_index = batch.edge_index.to(device, non_blocking=(device.type == "cuda"))
        edge_attr = batch.x_edge.to(device, non_blocking=(device.type == "cuda"))
        b = batch.node_graph_id.to(device, non_blocking=(device.type == "cuda"))
        return (x, edge_index, edge_attr, b), {}

    def build_prediction_fragment(
        self,
        *,
        batch,
        model_output,
        cfg: dict,
    ) -> dict:
        _ = cfg
        logits = model_output
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        probs_np = torch.sigmoid(logits).detach().cpu().to(torch.float32).numpy().astype("float32", copy=False)
        graph_event_ids_np = batch.graph_event_id.to(torch.int64).cpu().numpy().astype("int64", copy=False)
        graph_time_group_ids_np = batch.graph_time_group_id.to(torch.int64).cpu().numpy().astype("int64", copy=False)
        return {
            "prediction_event_ids_np": graph_event_ids_np,
            "model_outputs_by_name": {"main": probs_np},
            "time_group_event_ids_np": graph_event_ids_np,
            "time_group_ids_np": graph_time_group_ids_np,
        }


@step(name="run_group_classifier_inference", enable_cache=False)
def run_group_classifier_inference_step(
    model_info: dict,
    inputs: dict,
    writer_info: dict,
    loader_info: dict,
    pipeline_config: dict | None = None,
) -> dict:
    return GroupClassifierInferenceRunStep(pipeline_config=pipeline_config).execute(
        inputs=inputs,
        payloads={
            "loader_factory_init": loader_info,
            "writer_factory_init": writer_info,
            "model_handle_builder": model_info,
        },
    )
