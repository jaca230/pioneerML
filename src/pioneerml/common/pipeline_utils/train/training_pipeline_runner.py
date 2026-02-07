from __future__ import annotations


class TrainingPipelineRunner:
    """Shared step sequencing for training pipelines."""

    def run(
        self,
        *,
        pipeline_config,
        load_dataset_fn,
        tune_fn,
        train_fn,
        evaluate_fn,
        export_fn,
        load_kwargs: dict,
    ):
        if pipeline_config is not None and not isinstance(pipeline_config, dict):
            raise TypeError(f"Expected dict for pipeline_config, got {type(pipeline_config).__name__}.")

        dataset = load_dataset_fn(pipeline_config=pipeline_config, **load_kwargs)
        hpo_params = tune_fn(dataset, pipeline_config=pipeline_config)
        module = train_fn(dataset, pipeline_config=pipeline_config, hpo_params=hpo_params)
        metrics = evaluate_fn(module, dataset, pipeline_config=pipeline_config)
        export = export_fn(
            module,
            dataset,
            pipeline_config=pipeline_config,
            hpo_params=hpo_params,
            metrics=metrics,
        )
        return module, dataset, metrics, export
