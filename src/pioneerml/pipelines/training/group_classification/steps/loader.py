import torch
from torch_geometric.data import Data
from zenml import step

import pioneerml_dataloaders_python as pml
from pioneerml.common.pipeline_utils.loader import GraphDatasetBuilder
from pioneerml.common.zenml.materializers import GroupClassifierDatasetMaterializer
from pioneerml.pipelines.training.group_classification.dataset import GroupClassifierDataset
from pioneerml.pipelines.training.group_classification.steps.config import resolve_step_config


_DATASET_BUILDER = GraphDatasetBuilder()


def load_group_classifier_dataset_local(
    parquet_paths,
    config_json=None,
    normalize: bool = False,
    normalize_eps: float = 1e-6,
) -> GroupClassifierDataset:
    adapter = pml.adapters.input.graph.GroupClassifierInputAdapter()
    bundle = _DATASET_BUILDER.load_training_bundle(
        adapter,
        parquet_paths,
        config_json=config_json,
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
    u = _DATASET_BUILDER.arrow_to_torch(inputs.u, shape=(num_graphs, 1), dtype=torch.float32)
    time_group_ids = _DATASET_BUILDER.arrow_to_torch(inputs.time_group_ids, shape=(num_nodes,), dtype=torch.int64)
    graph_event_ids = _DATASET_BUILDER.arrow_to_torch(inputs.graph_event_ids, shape=(num_graphs,), dtype=torch.int64)
    graph_group_ids = _DATASET_BUILDER.arrow_to_torch(inputs.graph_group_ids, shape=(num_graphs,), dtype=torch.int64)
    batch = _DATASET_BUILDER.node_ptr_to_batch(node_ptr)

    if normalize:
        x_feat = x[:, :3]
        x_mean = x_feat.mean(dim=0)
        x_std = x_feat.std(dim=0).clamp_min(normalize_eps)
        x[:, :3] = (x_feat - x_mean) / x_std

        edge_feat = edge_attr[:, :3]
        edge_mean = edge_feat.mean(dim=0)
        edge_std = edge_feat.std(dim=0).clamp_min(normalize_eps)
        edge_attr[:, :3] = (edge_feat - edge_mean) / edge_std

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        u=u,
        time_group_ids=time_group_ids,
    )
    data.batch = batch
    data.num_graphs = num_graphs
    data.num_groups = num_groups
    data.graph_event_ids = graph_event_ids
    data.graph_group_ids = graph_group_ids
    data.group_ptr = group_ptr
    data.node_ptr = node_ptr
    data.edge_ptr = edge_ptr

    y = _DATASET_BUILDER.arrow_to_torch(targets.y, shape=(num_groups, 3), dtype=torch.float32)
    data.y = y
    return GroupClassifierDataset(data=data, targets=y)


@step(enable_cache=False, output_materializers=GroupClassifierDatasetMaterializer)
def load_group_classifier_dataset(
    parquet_paths: list[str],
    pipeline_config: dict | None = None,
) -> GroupClassifierDataset:
    step_config = resolve_step_config(pipeline_config, "loader")
    config_json = None if step_config is None else step_config.get("config_json")
    normalize = bool(step_config.get("normalize", False)) if step_config else False
    normalize_eps = float(step_config.get("normalize_eps", 1e-6)) if step_config else 1e-6
    return load_group_classifier_dataset_local(
        parquet_paths,
        config_json=config_json,
        normalize=normalize,
        normalize_eps=normalize_eps,
    )
