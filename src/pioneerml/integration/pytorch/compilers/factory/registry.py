from __future__ import annotations

from pioneerml.plugin import NamespacedPluginRegistry

from ..base_compiler import BaseCompiler

REGISTRY = NamespacedPluginRegistry[type[BaseCompiler]](
    namespace="compiler",
    expected_type=BaseCompiler,
    label="Compiler plugin",
)

