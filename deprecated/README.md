# Deprecated Code

This folder contains deprecated code that has been replaced by the new `src/pioneerml/` package structure.

## Contents

- `graph_data/` - Original package structure (replaced by `src/pioneerml/`)
  - `models.py` → `src/pioneerml/models/architectures.py`
  - `utils.py` → `src/pioneerml/data/datasets.py`

## Note

This code is kept for reference but should not be used in new development.
All functionality has been migrated to the new package structure.

If you're still using imports from `graph_data`, please update to:
```python
# Old
from graph_data.models import GroupClassifier
from graph_data.utils import GraphGroupDataset

# New
from pioneerml.models import GroupClassifier
from pioneerml.data import GraphGroupDataset
```
