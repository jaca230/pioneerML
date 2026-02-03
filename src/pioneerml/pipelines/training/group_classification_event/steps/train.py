import warnings
from typing import Mapping

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data
from zenml import step

from pioneerml.common.models.classifiers import GroupClassifierEvent
from pioneerml.common.training.lightning import GraphLightningModule
from pioneerml.common.zenml.utils import detect_available_accelerator

from pioneerml.pipelines.training.group_classification_event.dataset import GroupClassifierEventDataset
from pioneerml.pipelines.training.group_classification_event.steps.config import resolve_step_config


def _apply_lightning_warnings_filter() -> None:
    warnings.filterwarnings(
        "ignore",
        message="The 'train_dataloader' does not have many workers.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="isinstance\\(treespec, LeafSpec\\) is deprecated.*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="isinstance\\(treespec, LeafSpec\\) is deprecated.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="isinstance\\(treespec, LeafSpec\\) is deprecated.*",
        category=Warning,
        module="pytorch_lightning\\.utilities\\._pytree",
    )
    warnings.filterwarnings(
        "ignore",
        message="The 'val_dataloader' does not have many workers.*",
        category=UserWarning,
    )


def _merge_config(
    base: dict,
    override: Mapping | None,
) -> dict:
    if override is None:
        return base
    merged = dict(base)
    merged.update({k: v for k, v in override.items() if v is not None})
    return merged


def _split_dataset_to_graphs(dataset: GroupClassifierEventDataset) -> list[Data]:
    data = dataset.data
    if not hasattr(data, "node_ptr") or not hasattr(data, "edge_ptr") or not hasattr(data, "group_ptr"):
        raise AttributeError("Dataset data is missing node_ptr/edge_ptr/group_ptr required for batching.")
    node_ptr = data.node_ptr
    edge_ptr = data.edge_ptr
    group_ptr = data.group_ptr
    y = dataset.targets
    graphs: list[Data] = []
    num_graphs = int(getattr(data, "num_graphs", int(node_ptr.numel() - 1)))
    for g in range(num_graphs):
        node_start = int(node_ptr[g].item())
        node_end = int(node_ptr[g + 1].item())
        edge_start = int(edge_ptr[g].item())
        edge_end = int(edge_ptr[g + 1].item())
        group_start = int(group_ptr[g].item())
        group_end = int(group_ptr[g + 1].item())
        if node_end <= node_start or group_end <= group_start:
            continue

        x = data.x[node_start:node_end]
        edge_index_slice = data.edge_index[:, edge_start:edge_end]
        edge_attr = data.edge_attr[edge_start:edge_end]
        time_group_ids_slice = data.time_group_ids[node_start:node_end]

        edge_index = edge_index_slice
        if edge_index_slice.numel() > 0:
            edge_index = edge_index_slice - node_start

        time_group_ids = time_group_ids_slice
        if time_group_ids_slice.numel() > 0:
            min_gid = int(time_group_ids_slice.min().item())
            max_gid = int(time_group_ids_slice.max().item())
            if min_gid >= group_start and max_gid < group_end:
                time_group_ids = time_group_ids_slice - group_start
            elif min_gid >= 0 and max_gid < (group_end - group_start):
                time_group_ids = time_group_ids_slice
            else:
                time_group_ids = time_group_ids_slice - group_start
        y_slice = y[group_start:group_end]

        graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            time_group_ids=time_group_ids,
        )
        graph.y = y_slice
        graph.num_groups = int(group_end - group_start)
        graph.num_graphs = 1
        graphs.append(graph)
    return graphs


def _collate_graphs(items: list[Data]) -> Batch:
    batch = Batch.from_data_list(items)
    group_counts = [int(getattr(d, "num_groups", 0)) for d in items]
    total = 0
    ptr = [0]
    for count in group_counts:
        total += count
        ptr.append(total)
    batch.group_ptr = torch.tensor(ptr, dtype=torch.long, device=batch.x.device)
    batch.num_groups = int(total)
    return batch


@step
def train_group_classifier_event(
    dataset: GroupClassifierEventDataset,
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
        "threshold": 0.5,
        "trainer_kwargs": {"enable_progress_bar": True},
        "batch_size": 1,
        "shuffle": True,
        "model": {
            "in_dim": 4,
            "edge_dim": 4,
            "hidden": 200,
            "heads": 4,
            "num_blocks": 2,
            "dropout": 0.1,
        },
    }
    cfg = _merge_config(defaults, resolve_step_config(pipeline_config, "train"))
    if hpo_params:
        cfg = _merge_config(cfg, hpo_params)

    model_cfg = dict(cfg.get("model") or {})
    if "in_dim" not in model_cfg:
        model_cfg["in_dim"] = int(dataset.data.x.shape[-1])
    if "edge_dim" not in model_cfg:
        model_cfg["edge_dim"] = int(dataset.data.edge_attr.shape[-1])

    hidden = int(model_cfg.get("hidden", 200))
    heads = int(model_cfg.get("heads", 4))
    if hidden % heads != 0:
        raise ValueError(f"hidden ({hidden}) must be divisible by heads ({heads})")

    model = GroupClassifierEvent(
        in_dim=int(model_cfg["in_dim"]),
        edge_dim=int(model_cfg["edge_dim"]),
        hidden=hidden,
        heads=heads,
        num_blocks=int(model_cfg.get("num_blocks", 2)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        num_classes=int(dataset.targets.shape[-1]),
    )

    module = GraphLightningModule(
        model,
        task="classification",
        loss_fn=nn.BCEWithLogitsLoss(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
        threshold=float(cfg.get("threshold", 0.5)),
        scheduler_step_size=int(cfg["scheduler_step_size"]) if cfg.get("scheduler_step_size") is not None else None,
        scheduler_gamma=float(cfg["scheduler_gamma"]),
    )

    accelerator, devices = detect_available_accelerator()

    graphs = _split_dataset_to_graphs(dataset)
    if not graphs:
        raise RuntimeError("No non-empty graphs found in dataset for training.")

    train_loader = DataLoader(
        graphs,
        batch_size=int(cfg.get("batch_size", 1)),
        shuffle=bool(cfg.get("shuffle", True)),
        collate_fn=_collate_graphs,
    )
    val_loader = DataLoader(
        graphs,
        batch_size=int(cfg.get("batch_size", 1)),
        shuffle=False,
        collate_fn=_collate_graphs,
    )

    trainer_kwargs = dict(cfg.get("trainer_kwargs") or {})
    if cfg.get("grad_clip") is not None:
        trainer_kwargs.setdefault("gradient_clip_val", float(cfg["grad_clip"]))

    lightning_trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=int(cfg["max_epochs"]),
        enable_checkpointing=False,
        logger=False,
        **trainer_kwargs,
    )
    lightning_trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return module
