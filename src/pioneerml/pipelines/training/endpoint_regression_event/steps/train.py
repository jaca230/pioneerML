import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data
from zenml import step

from pioneerml.common.models.regressors import EndpointsRegressorEvent
from pioneerml.common.pipeline_utils.misc import LightningWarningFilter
from pioneerml.common.pipeline_utils.train import ModelTrainer
from pioneerml.common.training.lightning import GraphLightningModule
from pioneerml.pipelines.training.endpoint_regression_event.dataset import EndpointRegressorEventDataset
from pioneerml.pipelines.training.endpoint_regression_event.steps.config import resolve_step_config


_WARNING_FILTER = LightningWarningFilter()
_MODEL_TRAINER = ModelTrainer()


def _apply_lightning_warnings_filter() -> None:
    _WARNING_FILTER.apply_default()


def _merge_config(base: dict, override) -> dict:
    merged = dict(base)
    if override is not None:
        merged.update({k: v for k, v in override.items() if v is not None})
    return merged


def _split_dataset_to_graphs(dataset: EndpointRegressorEventDataset):
    data = dataset.data
    if not hasattr(data, "node_ptr") or not hasattr(data, "edge_ptr") or not hasattr(data, "group_ptr"):
        raise AttributeError("Dataset data is missing node_ptr/edge_ptr/group_ptr required for batching.")

    node_ptr = data.node_ptr
    edge_ptr = data.edge_ptr
    group_ptr = data.group_ptr
    y = dataset.targets
    num_graphs = int(getattr(data, "num_graphs", int(node_ptr.numel() - 1)))

    graphs: list[Data] = []
    for graph_idx in range(num_graphs):
        node_start = int(node_ptr[graph_idx].item())
        node_end = int(node_ptr[graph_idx + 1].item())
        edge_start = int(edge_ptr[graph_idx].item())
        edge_end = int(edge_ptr[graph_idx + 1].item())
        group_start = int(group_ptr[graph_idx].item())
        group_end = int(group_ptr[graph_idx + 1].item())

        if node_end <= node_start or group_end <= group_start:
            continue

        x = data.x[node_start:node_end]
        edge_index = data.edge_index[:, edge_start:edge_end]
        if edge_index.numel() > 0:
            edge_index = edge_index - node_start
        edge_attr = data.edge_attr[edge_start:edge_end]

        time_group_ids = data.time_group_ids[node_start:node_end]
        if time_group_ids.numel() > 0:
            min_gid = int(time_group_ids.min().item())
            max_gid = int(time_group_ids.max().item())
            if min_gid >= group_start and max_gid < group_end:
                time_group_ids = time_group_ids - group_start
            elif not (min_gid >= 0 and max_gid < (group_end - group_start)):
                time_group_ids = time_group_ids - group_start

        num_groups = int(group_end - group_start)
        graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y[group_start:group_end],
            time_group_ids=time_group_ids,
            group_probs=data.group_probs[group_start:group_end],
            splitter_probs=data.splitter_probs[node_start:node_end],
            group_ptr=torch.tensor([0, num_groups], dtype=torch.long),
        )
        graph.num_groups = num_groups
        if hasattr(data, "graph_event_ids"):
            graph.graph_event_id = data.graph_event_ids[graph_idx : graph_idx + 1]
        graphs.append(graph)

    return graphs


def _collate_graphs(items):
    batch = Batch.from_data_list(items)
    group_counts = [int(getattr(item, "num_groups", 0)) for item in items]
    total = 0
    ptr = [0]
    for count in group_counts:
        total += count
        ptr.append(total)
    batch.group_ptr = torch.tensor(ptr, dtype=torch.long, device=batch.x.device)
    batch.num_groups = int(total)
    return batch


@step
def train_endpoint_regressor_event(
    dataset: EndpointRegressorEventDataset,
    pipeline_config: dict | None = None,
    hpo_params: dict | None = None,
) -> GraphLightningModule:
    _apply_lightning_warnings_filter()

    defaults = {
        "max_epochs": 10,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "grad_clip": 2.0,
        "scheduler_step_size": 2,
        "scheduler_gamma": 0.5,
        "trainer_kwargs": {"enable_progress_bar": True},
        "batch_size": 1,
        "shuffle": True,
        "model": {
            "in_channels": 4,
            "group_prob_dimension": 3,
            "splitter_prob_dimension": 3,
            "hidden": 192,
            "heads": 4,
            "layers": 3,
            "dropout": 0.1,
            "output_dim": 6,
        },
    }
    cfg = _merge_config(defaults, resolve_step_config(pipeline_config, "train"))
    if hpo_params:
        cfg = _merge_config(cfg, hpo_params)

    model_cfg = dict(cfg.get("model") or {})
    if "in_channels" not in model_cfg:
        model_cfg["in_channels"] = int(dataset.data.x.shape[-1])
    if "group_prob_dimension" not in model_cfg:
        model_cfg["group_prob_dimension"] = int(dataset.data.group_probs.shape[-1])
    if "splitter_prob_dimension" not in model_cfg:
        model_cfg["splitter_prob_dimension"] = int(dataset.data.splitter_probs.shape[-1])

    hidden = int(model_cfg.get("hidden", 192))
    heads = int(model_cfg.get("heads", 4))
    if hidden % heads != 0:
        raise ValueError(f"hidden ({hidden}) must be divisible by heads ({heads})")

    model = EndpointsRegressorEvent(
        in_channels=int(model_cfg["in_channels"]),
        group_prob_dimension=int(model_cfg["group_prob_dimension"]),
        splitter_prob_dimension=int(model_cfg["splitter_prob_dimension"]),
        hidden=hidden,
        heads=heads,
        layers=int(model_cfg.get("layers", 3)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        output_dim=int(model_cfg.get("output_dim", dataset.targets.shape[-1])),
    )

    module = GraphLightningModule(
        model,
        task="regression",
        loss_fn=nn.MSELoss(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
        scheduler_step_size=int(cfg["scheduler_step_size"]) if cfg.get("scheduler_step_size") is not None else None,
        scheduler_gamma=float(cfg["scheduler_gamma"]),
    )

    graphs = _split_dataset_to_graphs(dataset)
    return _MODEL_TRAINER.fit(
        module=module,
        graphs=graphs,
        max_epochs=int(cfg["max_epochs"]),
        batch_size=int(cfg.get("batch_size", 1)),
        shuffle=bool(cfg.get("shuffle", True)),
        grad_clip=float(cfg["grad_clip"]) if cfg.get("grad_clip") is not None else None,
        trainer_kwargs=dict(cfg.get("trainer_kwargs") or {}),
        loader_cls=DataLoader,
        collate_fn=_collate_graphs,
    )
