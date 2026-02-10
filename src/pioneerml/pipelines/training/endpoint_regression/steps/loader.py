import torch
from torch_geometric.data import Data
from zenml import step

import pioneerml_dataloaders_python as pml
from pioneerml.common.pipeline_utils.loader import GraphDatasetBuilder
from pioneerml.common.zenml.materializers import EndpointRegressorDatasetMaterializer
from pioneerml.pipelines.training.endpoint_regression.dataset import EndpointRegressorDataset
from pioneerml.pipelines.training.endpoint_regression.steps.config import resolve_step_config


_DATASET_BUILDER = GraphDatasetBuilder()


def load_endpoint_regressor_dataset_local(
    parquet_paths,
    group_probs_parquet_paths=None,
    group_splitter_parquet_paths=None,
    config_json=None,
    normalize: bool = False,
    normalize_eps: float = 1e-6,
) -> EndpointRegressorDataset:
    adapter = pml.adapters.input.graph.EndpointRegressorInputAdapter()
    bundle = _DATASET_BUILDER.load_training_bundle(
        adapter,
        parquet_paths,
        config_json=config_json,
        secondary_key="group_probs",
        secondary_parquet_paths=group_probs_parquet_paths,
        extra_parquet_paths_by_key=(
            {"group_splitter_probs": group_splitter_parquet_paths}
            if group_splitter_parquet_paths is not None
            else None
        ),
    )
    inputs = bundle.inputs
    targets = bundle.targets

    num_graphs = int(inputs.num_graphs)
    num_groups = int(inputs.num_groups)
    node_ptr = _DATASET_BUILDER.arrow_to_torch(inputs.node_ptr, shape=(num_graphs + 1,), dtype=torch.int64)
    edge_ptr = _DATASET_BUILDER.arrow_to_torch(inputs.edge_ptr, shape=(num_graphs + 1,), dtype=torch.int64)
    group_ptr = _DATASET_BUILDER.arrow_to_torch(inputs.group_ptr, shape=(num_graphs + 1,), dtype=torch.int64)
    num_nodes = int(node_ptr[-1].item()) if num_graphs > 0 else 0
    num_edges = int(edge_ptr[-1].item()) if num_graphs > 0 else 0

    x = _DATASET_BUILDER.arrow_to_torch(inputs.node_features, shape=(num_nodes, 4), dtype=torch.float32)
    edge_index_pairs = _DATASET_BUILDER.arrow_to_torch(inputs.edge_index, shape=(num_edges, 2), dtype=torch.int64)
    edge_index = edge_index_pairs.t().contiguous()
    edge_attr = _DATASET_BUILDER.arrow_to_torch(inputs.edge_attr, shape=(num_edges, 4), dtype=torch.float32)
    time_group_ids = _DATASET_BUILDER.arrow_to_torch(inputs.time_group_ids, shape=(num_nodes,), dtype=torch.int64)
    u = _DATASET_BUILDER.arrow_to_torch(inputs.u, shape=(num_graphs, 1), dtype=torch.float32)
    group_probs = _DATASET_BUILDER.arrow_to_torch(inputs.group_probs, shape=(num_groups, 3), dtype=torch.float32)
    splitter_probs = _DATASET_BUILDER.arrow_to_torch(inputs.splitter_probs, shape=(num_nodes, 3), dtype=torch.float32)
    graph_event_ids = _DATASET_BUILDER.arrow_to_torch(inputs.graph_event_ids, shape=(num_graphs,), dtype=torch.int64)
    graph_group_ids = _DATASET_BUILDER.arrow_to_torch(inputs.graph_group_ids, shape=(num_graphs,), dtype=torch.int64)

    batch = _DATASET_BUILDER.node_ptr_to_batch(node_ptr)

    if normalize:
        if x.numel() > 0:
            x_feat = x[:, :3]
            x_mean = x_feat.mean(dim=0)
            x_std = x_feat.std(dim=0).clamp_min(normalize_eps)
            x[:, :3] = (x_feat - x_mean) / x_std

        if edge_attr.numel() > 0:
            edge_feat = edge_attr[:, :3]
            edge_mean = edge_feat.mean(dim=0)
            edge_std = edge_feat.std(dim=0).clamp_min(normalize_eps)
            edge_attr[:, :3] = (edge_feat - edge_mean) / edge_std

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        time_group_ids=time_group_ids,
        u=u,
        group_probs=group_probs,
        splitter_probs=splitter_probs,
    )
    data.batch = batch
    data.num_graphs = num_graphs
    data.num_groups = num_groups
    data.group_ptr = group_ptr
    data.node_ptr = node_ptr
    data.edge_ptr = edge_ptr
    data.graph_event_ids = graph_event_ids
    data.graph_group_ids = graph_group_ids

    # Time-group loader yields one target row per graph.
    y = _DATASET_BUILDER.arrow_to_torch(targets.y, shape=(num_graphs, 6), dtype=torch.float32)
    data.y = y
    return EndpointRegressorDataset(data=data, targets=y)


@step(enable_cache=False, output_materializers=EndpointRegressorDatasetMaterializer)
def load_endpoint_regressor_dataset(
    parquet_paths: list[str],
    group_probs_parquet_paths: list[str] | None = None,
    group_splitter_parquet_paths: list[str] | None = None,
    pipeline_config: dict | None = None,
) -> EndpointRegressorDataset:
    step_config = resolve_step_config(pipeline_config, "loader")
    config_json = None if step_config is None else step_config.get("config_json")
    normalize = bool(step_config.get("normalize", False)) if step_config else False
    normalize_eps = float(step_config.get("normalize_eps", 1e-6)) if step_config else 1e-6
    return load_endpoint_regressor_dataset_local(
        parquet_paths,
        group_probs_parquet_paths=group_probs_parquet_paths,
        group_splitter_parquet_paths=group_splitter_parquet_paths,
        config_json=config_json,
        normalize=normalize,
        normalize_eps=normalize_eps,
    )
