from zenml import pipeline

from .steps import (
    evaluate_group_classifier,
    export_group_classifier,
    load_group_classifier_data,
    train_group_classifier,
    tune_group_classifier,
)


def _get_step_config(config: dict | None, key: str) -> dict | None:
    if not config:
        return None
    raw = config.get(key)
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    raise TypeError(f"Expected dict for '{key}' config, got {type(raw).__name__}.")


@pipeline
def group_classification_pipeline(
    parquet_paths: list[str],
    *,
    pipeline_config: dict | None = None,
):
    loader_cfg = _get_step_config(pipeline_config, "loader")
    hpo_cfg = _get_step_config(pipeline_config, "hpo")
    train_cfg = _get_step_config(pipeline_config, "train")
    eval_cfg = _get_step_config(pipeline_config, "evaluate")

    batch = load_group_classifier_data(parquet_paths, step_config=loader_cfg)
    hpo_params = tune_group_classifier(batch, step_config=hpo_cfg)
    module = train_group_classifier(batch, step_config=train_cfg, hpo_params=hpo_params)
    metrics = evaluate_group_classifier(module, batch, step_config=eval_cfg)
    export = export_group_classifier(
        module,
        batch,
        step_config=_get_step_config(pipeline_config, "export"),
        hpo_params=hpo_params,
        metrics=metrics,
    )

    return module, batch, metrics, export
