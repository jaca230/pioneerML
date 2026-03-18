import torch
from zenml import step

from pioneerml.common.data_writer import TimeGroupPredictionSet
from pioneerml.common.pipeline.steps import BaseInferenceStep


class EndpointRegressorInferenceRunStep(BaseInferenceStep):
    step_key = "inference"
    prediction_set_cls = TimeGroupPredictionSet

    def default_config(self) -> dict:
        return {}

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

    def build_model_input(
        self,
        *,
        batch,
        device: torch.device,
        cfg: dict,
    ) -> tuple[tuple, dict]:
        _ = cfg
        return self._script_inputs(batch, device), {}

    def build_prediction_fragment(
        self,
        *,
        batch,
        model_output,
        cfg: dict,
    ) -> dict:
        _ = cfg
        preds = model_output
        preds = preds[0] if isinstance(preds, (tuple, list)) else preds
        preds_cpu = preds.detach().cpu().to(torch.float32)
        source_preds_np = preds_cpu.numpy().astype("float32", copy=False)
        graph_event_ids_np = batch.graph_event_id.to(torch.int64).cpu().numpy().astype("int64", copy=False)
        graph_time_group_ids_np = batch.graph_time_group_id.to(torch.int64).cpu().numpy().astype("int64", copy=False)
        return {
            "prediction_event_ids_np": graph_event_ids_np,
            "model_outputs_by_name": {"main": source_preds_np},
            "time_group_event_ids_np": graph_event_ids_np,
            "time_group_ids_np": graph_time_group_ids_np,
        }


@step(name="run_endpoint_regressor_inference", enable_cache=False)
def run_endpoint_regressor_inference_step(
    model_info: dict,
    inputs: dict,
    writer_info: dict,
    loader_info: dict,
    pipeline_config: dict | None = None,
) -> dict:
    return EndpointRegressorInferenceRunStep(pipeline_config=pipeline_config).execute(
        inputs=inputs,
        payloads={
            "loader_factory_init": loader_info,
            "writer_factory_init": writer_info,
            "model_handle_builder": model_info,
        },
    )
