from __future__ import annotations

from collections.abc import Mapping
import inspect
from pathlib import Path
from typing import Any

from pioneerml.integration.pytorch.model_handles import ModelHandleFactory

from .....resolver import BasePayloadResolver


class ModelHandleBuilderStateResolver(BasePayloadResolver):
    MODEL_GLOB_PATTERN_DEFAULT = "*_torchscript.pt"
    MODEL_REPO_PARENTS_UP_DEFAULT = 8
    MODEL_GLOB_BY_TYPE: dict[str, str] = {
        "script": "*_torchscript.pt",
        "torchscript": "*_torchscript.pt",
        "jit": "*_torchscript.pt",
        "trace": "*_trace.pt",
        "traced": "*_trace.pt",
        "export": "*.pt2",
        "torch_export": "*.pt2",
        "torchexport": "*.pt2",
    }

    def resolve(
        self,
        *,
        payloads: Mapping[str, Any] | None,
        runtime_state: dict[str, Any],
    ) -> None:
        selected = self._resolve_model_path_override(payloads=payloads)
        model_handle_block = dict(self.step.config_json.get("model_handle") or {})
        model_handle_cfg = dict(model_handle_block.get("config") or {})
        if selected is None:
            selected = model_handle_cfg.get("model_path")
        model_type = self._resolve_model_type(payloads=payloads)

        model_subdir = self._resolve_model_subdir()
        model_source_file = self._resolve_model_source_file()
        repo_parents_up = self._resolve_repo_parents_up()
        glob_pattern = self._resolve_model_glob_pattern(model_type=model_type)
        candidate_dirs = self._candidate_model_dirs(
            model_subdir=model_subdir,
            this_file=model_source_file,
            repo_parents_up=repo_parents_up,
        )
        resolved_model_path = self._resolve_model_path(
            selected_path=(None if selected is None else str(selected)),
            candidate_dirs=candidate_dirs,
            glob_pattern=glob_pattern,
        )
        runtime_state["resolved_model_path"] = resolved_model_path
        runtime_state["resolved_model_type"] = model_type
        runtime_state["model_handle"] = ModelHandleFactory(model_type=model_type).build(
            config={"model_path": resolved_model_path}
        )

    def _resolve_model_type(self, *, payloads: Mapping[str, Any] | None) -> str:
        if isinstance(payloads, Mapping):
            direct = payloads.get("model_type")
            if isinstance(direct, str) and direct.strip():
                return str(direct).strip().lower()
            payload = payloads.get("model_handle_builder") or payloads.get("model_handle_builder_payload")
            if isinstance(payload, Mapping):
                nested = payload.get("model_type")
                if isinstance(nested, str) and nested.strip():
                    return str(nested).strip().lower()
        model_handle_block = dict(self.step.config_json.get("model_handle") or {})
        raw = model_handle_block.get("type")
        if not isinstance(raw, str) or str(raw).strip() == "":
            raise RuntimeError(
                f"{self.step.__class__.__name__} requires non-empty 'model_handle.type' in config."
            )
        return str(raw).strip().lower()

    @staticmethod
    def _resolve_model_path_override(*, payloads: Mapping[str, Any] | None) -> str | None:
        if not isinstance(payloads, Mapping):
            return None

        direct = payloads.get("model_path")
        if isinstance(direct, str) and direct.strip():
            return str(direct)

        payload = payloads.get("model_handle_builder") or payloads.get("model_handle_builder_payload")
        if isinstance(payload, Mapping):
            nested = payload.get("model_path")
            if isinstance(nested, str) and nested.strip():
                return str(nested)

        return None

    def _resolve_model_subdir(self) -> str:
        model_handle_block = dict(self.step.config_json.get("model_handle") or {})
        model_handle_cfg = dict(model_handle_block.get("config") or {})
        out = str(model_handle_cfg.get("model_subdir", "")).strip()
        if not out:
            raise RuntimeError(
                f"{self.step.__class__.__name__} config missing non-empty 'model_handle.config.model_subdir'."
            )
        return out

    def _resolve_model_source_file(self) -> Path:
        fn = getattr(self.step, "model_source_file", None)
        if callable(fn):
            out = fn()
            if isinstance(out, Path):
                return out.resolve()
            if isinstance(out, str) and out.strip():
                return Path(out).resolve()
            raise TypeError(f"{self.step.__class__.__name__}.model_source_file() must return Path | str.")
        return Path(inspect.getfile(self.step.__class__)).resolve()

    def _resolve_repo_parents_up(self) -> int:
        value = getattr(self.step, "MODEL_REPO_PARENTS_UP", self.MODEL_REPO_PARENTS_UP_DEFAULT)
        return int(value)

    def _resolve_model_glob_pattern(self, *, model_type: str) -> str:
        value = getattr(self.step, "MODEL_GLOB_PATTERN", None)
        if value is None:
            value = self.MODEL_GLOB_BY_TYPE.get(str(model_type).strip().lower(), self.MODEL_GLOB_PATTERN_DEFAULT)
        out = str(value).strip()
        if not out:
            raise ValueError(f"{self.step.__class__.__name__} MODEL_GLOB_PATTERN cannot be empty.")
        return out

    @staticmethod
    def _candidate_model_dirs(*, model_subdir: str, this_file: Path, repo_parents_up: int) -> list[Path]:
        repo_root = this_file.parents[repo_parents_up] if len(this_file.parents) > repo_parents_up else this_file.parent
        cwd = Path.cwd().resolve()
        candidates = [
            cwd / "trained_models" / model_subdir,
            repo_root / "trained_models" / model_subdir,
            Path(f"/workspace/trained_models/{model_subdir}"),
        ]
        uniq: list[Path] = []
        seen: set[str] = set()
        for path in candidates:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(path)
        return uniq

    @staticmethod
    def _resolve_model_path(
        *,
        selected_path: str | None,
        candidate_dirs: list[Path],
        glob_pattern: str,
    ) -> str:
        if selected_path is not None:
            resolved = Path(selected_path).expanduser().resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Model not found: {resolved}")
            return str(resolved)

        candidates: list[Path] = []
        searched: list[str] = []
        for model_dir in candidate_dirs:
            searched.append(str(model_dir))
            if not model_dir.exists():
                continue
            candidates.extend(model_dir.glob(glob_pattern))
        candidates = sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError("No model files found. Searched directories: " + ", ".join(searched))
        return str(candidates[0].resolve())
