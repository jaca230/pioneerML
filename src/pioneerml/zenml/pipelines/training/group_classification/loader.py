from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import torch
from torch_geometric.data import Data

import pioneerml_dataloaders_python as pml


@dataclass
class GroupClassifierBatch:
    data: Data
    targets: torch.Tensor
    target_energy: torch.Tensor


def _capsule_to_array(arr):
    if isinstance(arr, tuple) and len(arr) == 2:
        schema_capsule, array_capsule = arr
        return pa.Array._import_from_c_capsule(schema_capsule, array_capsule)
    return arr


def _arrow_to_numpy(arr: pa.Array | pa.ChunkedArray | tuple) -> np.ndarray:
    arr = _capsule_to_array(arr)
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
        arr = arr.chunk(0)
    return arr.to_numpy(zero_copy_only=True)


def _ensure_writable(np_arr: np.ndarray) -> np.ndarray:
    if np_arr.flags.writeable:
        return np_arr
    try:
        np_arr.setflags(write=True)
        return np_arr
    except ValueError:
        return np_arr.copy()


def _arrow_to_torch(
    arr: pa.Array | pa.ChunkedArray,
    *,
    shape: Sequence[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    np_arr = _ensure_writable(_arrow_to_numpy(arr))
    return torch.from_numpy(np_arr).view(*shape).to(dtype)


def _node_ptr_to_batch(node_ptr: torch.Tensor) -> torch.Tensor:
    counts = (node_ptr[1:] - node_ptr[:-1]).to(torch.long)
    return torch.repeat_interleave(torch.arange(counts.numel(), device=counts.device), counts)


def _graph_u_to_node_u(u: torch.Tensor, node_ptr: torch.Tensor) -> torch.Tensor:
    counts = (node_ptr[1:] - node_ptr[:-1]).to(torch.long)
    return torch.repeat_interleave(u, counts, dim=0)


def _resolve_paths(paths: Iterable[str | Path]) -> list[str]:
    return [str(Path(p).expanduser().resolve()) for p in paths]


def load_group_classifier_batch(
    parquet_paths: Sequence[str | Path],
    *,
    config_json: Mapping | None = None,
) -> GroupClassifierBatch:
    adapter = pml.adapters.input.graph.GroupClassifierInputAdapter()
    if config_json is not None:
        adapter.load_config_json(json.dumps(config_json))

    paths = _resolve_paths(parquet_paths)
    bundle = adapter.load_training(paths)
    inputs = bundle.inputs
    targets = bundle.targets

    num_graphs = int(inputs.num_graphs)
    num_groups = int(inputs.num_groups)
    node_ptr = _arrow_to_torch(inputs.node_ptr, shape=(num_graphs + 1,), dtype=torch.int64)
    edge_ptr = _arrow_to_torch(inputs.edge_ptr, shape=(num_graphs + 1,), dtype=torch.int64)
    group_ptr = _arrow_to_torch(inputs.group_ptr, shape=(num_graphs + 1,), dtype=torch.int64)
    num_nodes = int(node_ptr[-1].item())
    num_edges = int(edge_ptr[-1].item())

    x = _arrow_to_torch(inputs.node_features, shape=(num_nodes, 4), dtype=torch.float32)
    edge_index = _arrow_to_torch(inputs.edge_index, shape=(2, num_edges), dtype=torch.int64)
    edge_attr = _arrow_to_torch(inputs.edge_attr, shape=(num_edges, 4), dtype=torch.float32)
    u = _arrow_to_torch(inputs.u, shape=(num_graphs, 1), dtype=torch.float32)
    time_group_ids = _arrow_to_torch(inputs.time_group_ids, shape=(num_nodes,), dtype=torch.int64)
    batch = _node_ptr_to_batch(node_ptr)
    u_nodes = _graph_u_to_node_u(u, node_ptr)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        u=u_nodes,
        time_group_ids=time_group_ids,
    )
    data.batch = batch
    data.num_graphs = num_graphs
    data.num_groups = num_groups
    data.graph_u = u
    data.group_ptr = group_ptr

    y = _arrow_to_torch(targets.y, shape=(num_groups, 3), dtype=torch.float32)
    y_energy = _arrow_to_torch(targets.y_energy, shape=(num_groups, 3), dtype=torch.float32)

    data.y = y
    return GroupClassifierBatch(data=data, targets=y, target_energy=y_energy)
