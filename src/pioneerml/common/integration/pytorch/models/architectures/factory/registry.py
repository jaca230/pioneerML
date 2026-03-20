from __future__ import annotations

from pioneerml.common.plugins import NamespacedPluginRegistry

from ..graph.base_graph_model import BaseGraphModel

REGISTRY = NamespacedPluginRegistry[type[BaseGraphModel]](
    namespace="architecture",
    expected_type=BaseGraphModel,
    label="Architecture plugin",
)
