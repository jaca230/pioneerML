from .builders import BasePluginBuilder, DefaultPluginBuilder

__all__ = [
    "BasePluginBuilder",
    "DefaultPluginBuilder",
    "NamespacedPluginRegistry",
    "PluginMappingView",
    "NamespacedPluginFactory",
]


def __getattr__(name: str):
    if name == "NamespacedPluginFactory":
        from .namespaced_factory import NamespacedPluginFactory

        return NamespacedPluginFactory
    if name in {"NamespacedPluginRegistry", "PluginMappingView"}:
        from .namespaced_registry import NamespacedPluginRegistry, PluginMappingView

        if name == "NamespacedPluginRegistry":
            return NamespacedPluginRegistry
        return PluginMappingView
    raise AttributeError(name)
