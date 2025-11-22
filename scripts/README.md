# Scripts

This directory contains example scripts for training and evaluation.

## Current Status

The training scripts are placeholders. For now, please use the Jupyter notebooks in the main repository:

- `classify_groups.ipynb` - Train the group classifier
- `group_splitter.ipynb` - Train the hit splitter
- `pion_stop.ipynb` - Train pion stop regression

## Planned Scripts

Future versions will include:

- `train_classifier.py` - Train group classifier from command line
- `train_splitter.py` - Train group splitter
- `train_pion_stop.py` - Train pion stop regressor
- `evaluate_model.py` - Evaluate trained models
- `run_inference.py` - Run inference on new data
- `preprocess_data.py` - Preprocess ROOT files to .npy format

## Usage Example (Future)

```bash
# Train classifier
python scripts/train_classifier.py \
    --data "/path/to/mainTimeGroups_*.npy" \
    --max-files 10 \
    --hidden 200 \
    --num-blocks 2 \
    --batch-size 128 \
    --epochs 20 \
    --output checkpoints/classifier.pt

# Evaluate
python scripts/evaluate_model.py \
    --checkpoint checkpoints/classifier.pt \
    --data "/path/to/test_data.npy"
```
