# Endpoint regressor (training/validation)

- Model: `OrthogonalEndpointRegressor` (`EndpointRegressor` alias) from `pioneerml.models.regressors.endpoint_regressor`.
- Input: raw hit graphs (`Data(x, edge_index, edge_attr)`) built from `hits_batch_*.npy` + `group_info_batch_*.npy` via `load_hits_and_info`.
- Output: `endpoint_quantiles` per view (and optional `group_score`/`group_energy`), log to `data/upstream_preds/`.
- Training: mirror the group classifier pipeline but use regression losses; skip UMAP/SNE/PCA; reuse regressor plots (loss curves, scatter, residuals).
- Validation: load saved checkpoint, run on held-out raw hits, write predictions to `data/upstream_preds/`, and plot regression diagnostics.
