from __future__ import annotations

from pioneerml.plugin import NamespacedPluginFactory

from .base_plot import BasePlot


class PlotFactory(NamespacedPluginFactory[BasePlot]):
    def __init__(
        self,
        *,
        plot_cls: type[BasePlot] | None = None,
        plot_name: str | None = None,
    ) -> None:
        super().__init__(
            namespace="plot",
            plugin_cls=plot_cls,
            plugin_name=plot_name,
            expected_instance_type=BasePlot,
            label="Plot",
        )
