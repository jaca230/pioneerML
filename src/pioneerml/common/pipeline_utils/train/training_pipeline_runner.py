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

        if load_dataset_fn is not None:
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

        hpo_params = tune_fn(pipeline_config=pipeline_config, **load_kwargs)
        module = train_fn(pipeline_config=pipeline_config, hpo_params=hpo_params, **load_kwargs)
        metrics = evaluate_fn(module, pipeline_config=pipeline_config, **load_kwargs)
        export = export_fn(
            module,
            pipeline_config=pipeline_config,
            hpo_params=hpo_params,
            metrics=metrics,
            **load_kwargs,
        )
        return module, None, metrics, export
