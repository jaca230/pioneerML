from __future__ import annotations

from dataclasses import replace

from pioneerml.common.loader.array_store.ndarray_store import NDArrayColumnSpec
from pioneerml.common.loader.config import DataFlowConfig, SplitSampleConfig
from pioneerml.common.parquet import ParquetChunkReader, ParquetInputSet

from ..base_loader import BaseLoader


class ParquetLoader(BaseLoader):
    """Loader constrained to parquet chunked inputs."""

    def __init__(
        self,
        *,
        parquet_inputs: ParquetInputSet,
        resolved_column_specs: tuple[NDArrayColumnSpec, ...] | None = None,
        mode: str | None = None,
        data_flow_config: DataFlowConfig | None = None,
        split_config: SplitSampleConfig | None = None,
    ) -> None:
        super().__init__(data_flow_config=data_flow_config, mode=mode)
        self.parquet_inputs = parquet_inputs
        self.parquet_paths = list(parquet_inputs.main_paths)
        self.row_groups_per_chunk = int(self.data_flow_config.row_groups_per_chunk)
        self.resolved_column_specs = tuple(
            resolved_column_specs if resolved_column_specs is not None else ()
        )
        self.input_columns = self.columns_from_specs(
            column_specs=self.resolved_column_specs,
            target_only=False,
            required=True,
        )
        self.target_columns = self.columns_from_specs(
            column_specs=self.resolved_column_specs,
            target_only=True,
            required=True,
        )
        self.optional_input_columns = self.columns_from_specs(
            column_specs=self.resolved_column_specs,
            target_only=False,
            required=False,
        )
        self.columns = self.columns_from_specs(
            column_specs=self.resolved_column_specs,
            source="main",
        )
        self.split_config = split_config if split_config is not None else SplitSampleConfig()
        self.split = self.split_config.split
        self.train_fraction = float(self.split_config.train_fraction)
        self.val_fraction = float(self.split_config.val_fraction)
        self.test_fraction = float(self.split_config.test_fraction)
        self.split_seed = int(self.split_config.split_seed if self.split_config.split_seed is not None else 0)
        self.sample_fraction = self.split_config.sample_fraction
        self.edge_populate_graph_block = 512

    @classmethod
    def resolve_dynamic_column_map(
        cls,
        *,
        parquet_inputs: ParquetInputSet,
        required_columns: list[str],
        optional_columns: list[str] | None = None,
    ) -> dict[str, str]:
        """
        Resolve a logical column->source plan without materializing joined tables.

        Source precedence is:
        1) main
        2) optional sources in insertion order of ParquetInputSet optional_paths_by_name.
        """
        resolved_optional_sources: dict[str, list[str]] = {
            str(name): list(paths) for name, paths in parquet_inputs.optional_paths_by_name.items()
        }

        available_by_source: dict[str, set[str]] = {
            "main": ParquetInputSet.schema_columns_intersection(parquet_inputs.main_paths)
        }
        for source_name, source_paths in resolved_optional_sources.items():
            available_by_source[source_name] = ParquetInputSet.schema_columns_intersection(tuple(source_paths))

        all_required = [str(c) for c in required_columns]
        all_optional = [str(c) for c in (optional_columns or [])]

        column_to_source: dict[str, str] = {}

        def assign_column(column_name: str, *, required: bool) -> None:
            if column_name in column_to_source:
                return
            if column_name in available_by_source["main"]:
                column_to_source[column_name] = "main"
                return
            for source_name in resolved_optional_sources:
                if column_name in available_by_source[source_name]:
                    column_to_source[column_name] = source_name
                    return
            if required:
                raise ValueError(f"Required column '{column_name}' not found in main or optional parquet sources.")

        for col in all_required:
            assign_column(col, required=True)
        for col in all_optional:
            assign_column(col, required=False)

        return column_to_source

    @classmethod
    def resolve_declared_columns_from_sources(
        cls,
        *,
        parquet_inputs: ParquetInputSet,
        column_specs: tuple[NDArrayColumnSpec, ...],
        include_targets: bool,
    ) -> dict[str, object]:
        required_inputs = [str(s.column) for s in column_specs if (not bool(s.target_only)) and bool(s.required)]
        required_targets = [str(s.column) for s in column_specs if bool(s.target_only) and bool(s.required)]
        optional_inputs = [str(s.column) for s in column_specs if (not bool(s.target_only)) and (not bool(s.required))]
        required_columns = list(required_inputs) + (list(required_targets) if include_targets else [])
        column_to_source = cls.resolve_dynamic_column_map(
            parquet_inputs=parquet_inputs,
            required_columns=required_columns,
            optional_columns=optional_inputs,
        )
        selected_columns = set(str(c) for c in cls.unique_columns(required_columns + optional_inputs))
        resolved_column_specs: list[NDArrayColumnSpec] = []
        for spec in column_specs:
            col = str(spec.column)
            if col not in selected_columns:
                continue
            if col not in column_to_source:
                continue
            resolved_column_specs.append(replace(spec, source=str(column_to_source[col])))
        return {"resolved_column_specs": tuple(resolved_column_specs)}

    @classmethod
    def columns_from_specs(
        cls,
        *,
        column_specs: tuple[NDArrayColumnSpec, ...],
        target_only: bool | None = None,
        required: bool | None = None,
        source: str | None = None,
    ) -> list[str]:
        cols: list[str] = []
        for spec in column_specs:
            if target_only is not None and bool(spec.target_only) != bool(target_only):
                continue
            if required is not None and bool(spec.required) != bool(required):
                continue
            if source is not None and str(spec.source) != str(source):
                continue
            cols.append(str(spec.column))
        return cls.unique_columns(cols)

    def required_columns(self, *, include_targets: bool | None = None) -> list[str]:
        use_targets = self.include_targets if include_targets is None else bool(include_targets)
        cols = [*self.input_columns]
        if use_targets:
            cols.extend(self.target_columns)
        return list(dict.fromkeys(cols))

    @staticmethod
    def unique_columns(columns: list[str]) -> list[str]:
        return list(dict.fromkeys(columns))

    def _iter_tables(self):
        reader = ParquetChunkReader(
            parquet_paths=self.parquet_paths,
            columns=self.columns,
            row_groups_per_chunk=self.row_groups_per_chunk,
        )
        yield from reader.iter_tables()
