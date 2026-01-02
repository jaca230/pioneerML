## Notebook update note

Training/validation notebooks should switch to the new paired hits/info loader that matches the updated GraphRecord schema.

Use:
```python
from pioneerml.data import load_hits_and_info

records = load_hits_and_info(
    hits_pattern="data/hits_batch_*.npy",
    info_pattern="data/group_info_batch_*.npy",
    include_hit_labels=True,  # set False if not needed
)
```

This loader:
- Parses per-hit coord/z/energy/view/pdg_mask.
- Emits group-level labels, energies, pion stop, angle vector, start/end; missing arc length defaults to 0.
- Returns `GraphRecord` objects compatible with the stereo-aware datasets/models.

For splitter/endpoint training, wrap the returned `records` with the appropriate dataset:
```python
from pioneerml.data import GraphGroupDataset, SplitterGraphDataset
ds = GraphGroupDataset(records, num_classes=3)
# or
splitter_ds = SplitterGraphDataset(records, use_group_probs=True)
```

Update any legacy `load_preprocessed_time_groups` calls to `load_hits_and_info` to stay aligned with the new data format.
