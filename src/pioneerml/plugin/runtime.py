from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
from typing import Iterable

_PLUGINS_LOADED = False


def _iter_repo_plugin_src_roots() -> Iterable[Path]:
    # src/pioneerml/plugin/runtime.py -> repo root
    repo_root = Path(__file__).resolve().parents[3]
    plugins_root = repo_root / "plugins"
    if not plugins_root.exists():
        return ()
    out: list[Path] = []
    for child in plugins_root.iterdir():
        if not child.is_dir():
            continue
        src_dir = child / "src"
        if src_dir.is_dir():
            out.append(src_dir)
    return out


def _inject_src_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        s = str(path.resolve())
        if s not in sys.path:
            sys.path.insert(0, s)


def ensure_plugins_loaded() -> None:
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return

    env_src_paths = os.environ.get("PIONEERML_PLUGIN_SRC_PATHS", "")
    extra_src_paths = [Path(p).expanduser().resolve() for p in env_src_paths.split(os.pathsep) if p.strip()]
    _inject_src_paths([*extra_src_paths, *_iter_repo_plugin_src_roots()])

    env_modules = os.environ.get("PIONEERML_PLUGIN_MODULES", "pioneerml_base_plugin")
    modules = [m.strip() for m in env_modules.split(",") if m.strip()]
    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue

    _PLUGINS_LOADED = True

