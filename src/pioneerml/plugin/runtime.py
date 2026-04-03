from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys
from dataclasses import dataclass
from typing import Iterable

_PLUGINS_LOADED = False
_MANIFEST_FILENAME = "plugin.json"


@dataclass(frozen=True)
class PluginManifest:
    plugin_dir: Path
    src_dir: Path
    name: str
    module: str
    version: str
    description: str
    depends_on: tuple[str, ...]


def _iter_repo_plugin_dirs() -> Iterable[Path]:
    # src/pioneerml/plugin/runtime.py -> repo root
    repo_root = Path(__file__).resolve().parents[3]
    plugins_root = repo_root / "plugins"
    if not plugins_root.exists():
        return ()
    out: list[Path] = []
    for child in plugins_root.iterdir():
        if child.is_dir():
            out.append(child)
    return tuple(sorted(out))


def _iter_repo_plugin_src_roots() -> Iterable[Path]:
    out: list[Path] = []
    for child in _iter_repo_plugin_dirs():
        src_dir = child / "src"
        if src_dir.is_dir():
            out.append(src_dir)
    return tuple(out)


def _inject_src_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        s = str(path.resolve())
        if s not in sys.path:
            sys.path.insert(0, s)


def _discover_modules_in_src_root(src_root: Path) -> tuple[str, ...]:
    modules: set[str] = set()
    if not src_root.is_dir():
        return ()
    for child in src_root.iterdir():
        if not child.is_dir():
            continue
        if not (child / "__init__.py").is_file():
            continue
        modules.add(str(child.name))
    return tuple(sorted(modules))


def _discover_plugin_modules(src_roots: Iterable[Path]) -> tuple[str, ...]:
    modules: set[str] = set()
    for src_root in src_roots:
        for module in _discover_modules_in_src_root(src_root):
            modules.add(module)
    return tuple(sorted(modules))


def _normalize_depends_on(*, depends_on: object, manifest_path: Path) -> tuple[str, ...]:
    if depends_on in (None, ""):
        return ()
    if not isinstance(depends_on, (list, tuple)):
        raise RuntimeError(
            f"Invalid plugin manifest '{manifest_path}': 'depends_on' must be a list of plugin names/modules."
        )
    out: list[str] = []
    for dep in depends_on:
        if not isinstance(dep, str) or dep.strip() == "":
            raise RuntimeError(
                f"Invalid plugin manifest '{manifest_path}': each 'depends_on' entry must be a non-empty string."
            )
        out.append(dep.strip())
    return tuple(out)


def _read_repo_plugin_manifests(plugin_dirs: Iterable[Path]) -> tuple[PluginManifest, ...]:
    manifests: list[PluginManifest] = []
    for plugin_dir in plugin_dirs:
        src_dir = plugin_dir / "src"
        if not src_dir.is_dir():
            continue
        manifest_path = plugin_dir / _MANIFEST_FILENAME
        if not manifest_path.is_file():
            continue
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Failed to parse plugin manifest '{manifest_path}': {exc}") from exc
        if not isinstance(raw, dict):
            raise RuntimeError(f"Invalid plugin manifest '{manifest_path}': top-level JSON value must be an object.")

        name = str(raw.get("name") or plugin_dir.name).strip()
        module = str(raw.get("module") or "").strip()
        if module == "":
            raise RuntimeError(
                f"Invalid plugin manifest '{manifest_path}': missing required field 'module'."
            )

        available_modules = _discover_modules_in_src_root(src_dir)
        if module not in available_modules:
            raise RuntimeError(
                f"Invalid plugin manifest '{manifest_path}': module '{module}' not found under '{src_dir}'. "
                f"Discovered: {', '.join(available_modules) if available_modules else '<none>'}"
            )

        version = str(raw.get("version") or "").strip()
        description = str(raw.get("description") or "").strip()
        depends_on = _normalize_depends_on(
            depends_on=raw.get("depends_on", raw.get("dependencies", [])),
            manifest_path=manifest_path,
        )
        manifests.append(
            PluginManifest(
                plugin_dir=plugin_dir,
                src_dir=src_dir,
                name=name,
                module=module,
                version=version,
                description=description,
                depends_on=depends_on,
            )
        )
    return tuple(manifests)


def _resolve_manifest_import_order(manifests: Iterable[PluginManifest]) -> tuple[str, ...]:
    manifests_by_module: dict[str, PluginManifest] = {}
    alias_to_module: dict[str, str] = {}
    indegree: dict[str, int] = {}
    dependents: dict[str, set[str]] = {}

    for manifest in manifests:
        module = manifest.module
        if module in manifests_by_module and manifests_by_module[module].plugin_dir != manifest.plugin_dir:
            prev = manifests_by_module[module]
            raise RuntimeError(
                f"Plugin module '{module}' declared in multiple manifests: "
                f"'{prev.plugin_dir / _MANIFEST_FILENAME}' and '{manifest.plugin_dir / _MANIFEST_FILENAME}'."
            )
        manifests_by_module[module] = manifest
        indegree[module] = 0
        dependents[module] = set()

    for manifest in manifests:
        alias_to_module[manifest.module] = manifest.module
        alias_to_module[manifest.name] = manifest.module

    for manifest in manifests:
        for dep in manifest.depends_on:
            dep_module = alias_to_module.get(dep)
            if dep_module is None:
                raise RuntimeError(
                    f"Plugin '{manifest.name}' declares missing dependency '{dep}'. "
                    f"Known plugins: {', '.join(sorted(alias_to_module.keys())) or '<none>'}"
                )
            if dep_module == manifest.module:
                continue
            if manifest.module not in dependents[dep_module]:
                dependents[dep_module].add(manifest.module)
                indegree[manifest.module] = indegree.get(manifest.module, 0) + 1

    ready = sorted([module for module, degree in indegree.items() if degree == 0])
    ordered: list[str] = []
    while len(ready) > 0:
        current = ready.pop(0)
        ordered.append(current)
        for dep_module in sorted(dependents[current]):
            indegree[dep_module] = indegree[dep_module] - 1
            if indegree[dep_module] == 0:
                ready.append(dep_module)
        ready.sort()

    if len(ordered) != len(manifests_by_module):
        unresolved = sorted([module for module, degree in indegree.items() if degree > 0])
        raise RuntimeError(
            "Plugin dependency cycle detected in manifests for modules: "
            + ", ".join(unresolved)
        )
    return tuple(ordered)


def _dedupe_preserve_order(values: Iterable[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return tuple(out)


def list_plugin_manifests() -> tuple[dict[str, object], ...]:
    """Return manifest metadata for repo plugins declaring `plugin.json`."""
    manifests = _read_repo_plugin_manifests(_iter_repo_plugin_dirs())
    return tuple(
        {
            "name": manifest.name,
            "module": manifest.module,
            "version": manifest.version,
            "description": manifest.description,
            "depends_on": manifest.depends_on,
            "plugin_dir": str(manifest.plugin_dir),
        }
        for manifest in manifests
    )


def ensure_plugins_loaded() -> None:
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return

    env_src_paths = os.environ.get("PIONEERML_PLUGIN_SRC_PATHS", "")
    extra_src_paths = [Path(p).expanduser().resolve() for p in env_src_paths.split(os.pathsep) if p.strip()]
    repo_plugin_dirs = list(_iter_repo_plugin_dirs())
    repo_src_roots = list(_iter_repo_plugin_src_roots())
    all_src_roots = [*extra_src_paths, *repo_src_roots]
    _inject_src_paths(all_src_roots)

    env_modules = os.environ.get("PIONEERML_PLUGIN_MODULES", "").strip()
    manifest_modules: tuple[str, ...] = ()
    if env_modules != "":
        modules = [m.strip() for m in env_modules.split(",") if m.strip()]
        required_modules = set(modules)
    else:
        manifests = _read_repo_plugin_manifests(repo_plugin_dirs)
        manifest_modules = _resolve_manifest_import_order(manifests)
        discovered_modules = _discover_plugin_modules(all_src_roots)
        modules = list(_dedupe_preserve_order((*manifest_modules, *discovered_modules)))
        required_modules = set(manifest_modules)

    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if module_name in required_modules:
                raise RuntimeError(
                    f"Failed to import required plugin module '{module_name}'. "
                    "Check plugin manifest dependencies and installation paths."
                ) from exc
            continue

    _PLUGINS_LOADED = True
