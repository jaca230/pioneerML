from __future__ import annotations

from collections.abc import Mapping

from pioneerml.plugin import NamespacedPluginRegistry

from .base_plot import BasePlot

REGISTRY = NamespacedPluginRegistry[type[BasePlot]](
    namespace="plot",
    expected_type=BasePlot,
    label="Plot plugin",
)

PLOT_REGISTRY: Mapping[str, type[BasePlot]] = REGISTRY.mapping_view()
