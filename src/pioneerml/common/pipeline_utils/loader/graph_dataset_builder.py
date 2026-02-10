from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import torch


class GraphDatasetBuilder:
    """Shared helpers for adapter-driven dataset loading and Arrow/Torch conversion."""

    def _capsule_to_array(self, arr):
        if isinstance(arr, tuple) and len(arr) == 2:
            schema_capsule, array_capsule = arr
            return pa.Array._import_from_c_capsule(schema_capsule, array_capsule)
        return arr

    def arrow_to_numpy(self, arr: pa.Array | pa.ChunkedArray | tuple) -> np.ndarray:
        arr = self._capsule_to_array(arr)
        if isinstance(arr, pa.ChunkedArray):
            arr = arr.combine_chunks().chunk(0)
        return arr.to_numpy(zero_copy_only=True)

    def ensure_writable(self, np_arr: np.ndarray) -> np.ndarray:
        if np_arr.flags.writeable:
            return np_arr
        try:
            np_arr.setflags(write=True)
            return np_arr
        except ValueError:
            return np_arr.copy()

    def arrow_to_torch(
        self,
        arr: pa.Array | pa.ChunkedArray | tuple,
        *,
        shape: Sequence[int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        np_arr = self.ensure_writable(self.arrow_to_numpy(arr))
        return torch.from_numpy(np_arr).view(*shape).to(dtype)

    def node_ptr_to_batch(self, node_ptr: torch.Tensor) -> torch.Tensor:
        counts = (node_ptr[1:] - node_ptr[:-1]).to(torch.long)
        return torch.repeat_interleave(torch.arange(counts.numel(), device=counts.device), counts)

    def resolve_paths(self, paths: Iterable[str | Path]) -> list[str]:
        return [str(Path(p).expanduser().resolve()) for p in paths]

    def build_input_spec(
        self,
        paths: Sequence[str],
        *,
        secondary_key: str | None = None,
        secondary_paths: Sequence[str] | None = None,
        extra_paths_by_key: Mapping[str, Sequence[str]] | None = None,
    ) -> dict:
        if secondary_paths is not None and len(secondary_paths) != len(paths):
            raise ValueError(
                f"secondary_paths must match paths length ({len(secondary_paths)} != {len(paths)})."
            )
        if extra_paths_by_key is not None:
            for key, extra_paths in extra_paths_by_key.items():
                if len(extra_paths) != len(paths):
                    raise ValueError(
                        f"extra_paths_by_key[{key!r}] length must match paths length "
                        f"({len(extra_paths)} != {len(paths)})."
                    )
        files = []
        for idx, main_path in enumerate(paths):
            item = {"mainFile": main_path}
            if secondary_key is not None and secondary_paths is not None:
                item[secondary_key] = secondary_paths[idx]
            if extra_paths_by_key is not None:
                for key, extra_paths in extra_paths_by_key.items():
                    item[key] = extra_paths[idx]
            files.append(item)
        return {"files": files}

    def load_training_bundle(
        self,
        adapter,
        parquet_paths: Sequence[str | Path],
        *,
        config_json: Mapping | None = None,
        secondary_key: str | None = None,
        secondary_parquet_paths: Sequence[str | Path] | None = None,
        extra_parquet_paths_by_key: Mapping[str, Sequence[str | Path]] | None = None,
    ):
        if config_json is not None:
            if hasattr(adapter, "load_config_json"):
                adapter.load_config_json(json.dumps(config_json))
            else:
                adapter.load_config(config_json)

        paths = self.resolve_paths(parquet_paths)
        secondary_paths = (
            self.resolve_paths(secondary_parquet_paths)
            if secondary_parquet_paths is not None
            else None
        )
        extra_paths_by_key = None
        if extra_parquet_paths_by_key is not None:
            extra_paths_by_key = {
                key: self.resolve_paths(extra_paths)
                for key, extra_paths in extra_parquet_paths_by_key.items()
            }
        spec = self.build_input_spec(
            paths,
            secondary_key=secondary_key,
            secondary_paths=secondary_paths,
            extra_paths_by_key=extra_paths_by_key,
        )
        return adapter.load_training_json(json.dumps(spec))
