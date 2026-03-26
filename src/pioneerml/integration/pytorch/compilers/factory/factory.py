from __future__ import annotations

from pioneerml.plugin import NamespacedPluginFactory

from ..base_compiler import BaseCompiler


class CompilerFactory(NamespacedPluginFactory[BaseCompiler]):
    def __init__(
        self,
        *,
        compiler_cls: type[BaseCompiler] | None = None,
        compiler_name: str | None = None,
    ) -> None:
        super().__init__(
            namespace="compiler",
            plugin_cls=compiler_cls,
            plugin_name=compiler_name,
            expected_instance_type=BaseCompiler,
            label="Compiler",
        )

