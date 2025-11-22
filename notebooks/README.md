# Notebooks

This directory contains Jupyter notebooks for training, evaluation, and examples.

## Structure

- `training/` - Training pipelines for different models
- `evaluation/` - Model evaluation and analysis notebooks
- `examples/` - Usage examples and tutorials

## Training Notebooks

### Current Models

- `classify_groups.ipynb` - Time group classification (pion/muon/MIP)
- `group_splitter.ipynb` - Multi-particle hit splitting
- `pion_stop.ipynb` - Pion stop position regression
- `endpoint_finder.ipynb` - Track endpoint prediction (in development)
- `positron_angle.ipynb` - Positron angle prediction (in development)

## Usage

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```

2. Navigate to the desired notebook

3. Run cells sequentially

## Note

These notebooks currently use the legacy `graph_data` imports. They will be updated to use the new `pioneerml` package structure in a future update.
