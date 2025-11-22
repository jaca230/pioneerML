# PIONEER ML

Machine learning pipeline framework for the PIONEER experiment's AI-based reconstruction system.

## Overview

PIONEER ML provides Graph Neural Network (GNN) models and training pipelines for reconstructing particle physics events from the PIONEER Active Target (ATAR) detector. The system processes time-grouped detector hits through multiple AI models to identify particles (pions, muons, positrons) and extract physics features.

### Key Features

- **Standardized Graph Neural Networks**: Transformer-based GNN architectures for various reconstruction tasks
- **Modular Pipeline Framework**: Composable stages for classification, splitting, and regression
- **Multiple Reconstruction Models**:
  - Time group classification (particle identification)
  - Multi-particle hit splitting
  - Pion stop position regression
  - Track endpoint finding
  - Positron angle prediction
- **Comprehensive Training Utilities**: Ready-to-use training loops with metrics and visualization

## Architecture

The reconstruction pipeline consists of multiple stages:

1. **Time Grouping**: Preprocess detector hits into time windows (1ns)
2. **Group Classification**: Identify particle types in each time group (pion/muon/MIP)
3. **Hit Splitting**: Assign individual hits to particles in multi-particle groups
4. **Feature Extraction**: Predict physics quantities (positions, angles, energies)
5. **Pattern Matching**: Connect time groups into continuous tracks

All models use a standardized graph representation:
- **Node features** (5D): `[coord, z, energy, view, group_energy]`
- **Edge features** (4D): `[dx, dz, dE, same_view]`
- **Architecture**: Transformer-based with JumpingKnowledge and attentional pooling

## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended for training)
- Git

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone git@github.com:jaca230/pioneerML.git
   cd pioneerML
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   # Core dependencies
   pip install -r requirements.txt

   # Or for development (includes testing, formatting tools)
   pip install -r requirements-dev.txt

   # Or install as editable package
   pip install -e .
   ```

4. **Install PyTorch Geometric dependencies**:

   PyTorch Geometric requires additional packages. Install them based on your PyTorch and CUDA versions:

   ```bash
   # For CUDA 11.8 (adjust based on your setup)
   pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

   # See https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
   ```

### Optional Dependencies

- **PyTorch Lightning** (for advanced training features):
  ```bash
  pip install pytorch-lightning tensorboard
  ```

- **Experiment Tracking** (Weights & Biases or MLflow):
  ```bash
  pip install wandb
  # or
  pip install mlflow
  ```

- **ROOT/PyROOT** (for reading CERN ROOT files):
  ```bash
  # Install via conda (recommended)
  conda install -c conda-forge root
  ```

## Usage

### Training a Model

The repository includes Jupyter notebooks for training each model:

- `classify_groups.ipynb` - Train the group classifier
- `group_splitter.ipynb` - Train the hit splitter
- `pion_stop.ipynb` - Train pion stop regression
- `endpoint_finder.ipynb` - Track endpoint prediction (in development)
- `positron_angle.ipynb` - Positron angle prediction (in development)

### Example: Group Classification

```python
from graph_data.models import GroupClassifier
from graph_data.utils import GraphGroupDataset

# Load preprocessed data
from your_loader import load_preprocessed_time_groups

groups = load_preprocessed_time_groups(
    '/path/to/data/mainTimeGroups_*.npy',
    max_files=4,
    limit_groups=100000,
    min_hits_per_label=2,
)

# Create dataset
dataset = GraphGroupDataset(groups, num_classes=3)

# Initialize model
model = GroupClassifier(
    hidden=200,
    num_blocks=2,
    heads=4,
    dropout=0.05
)

# Train (see notebooks for complete training functions)
```

### Using Pre-trained Models

```python
import torch
from graph_data.models import GroupClassifier

# Load checkpoint
model = GroupClassifier(hidden=200, num_blocks=2)
model.load_state_dict(torch.load('checkpoints/classifier.pt'))
model.eval()

# Inference
with torch.no_grad():
    predictions = model(data)
```

## Project Structure

```
pioneerML/
├── graph_data/              # Core Python package
│   ├── models.py           # GNN model definitions
│   └── utils.py            # Dataset classes and utilities
├── legacy_networks/         # Archived implementations
├── classify_groups.ipynb   # Group classification training
├── group_splitter.ipynb    # Multi-particle splitting training
├── pion_stop.ipynb         # Pion stop regression training
├── endpoint_finder.ipynb   # Endpoint finding (in development)
├── positron_angle.ipynb    # Positron angle (in development)
├── pyproject.toml          # Project configuration
├── requirements.txt        # Core dependencies
├── requirements-dev.txt    # Development dependencies
└── README.md               # This file
```

## Models

### 1. GroupClassifier
**Purpose**: Multi-label classification of time groups
**Classes**: Pion (0), Muon (1), MIP/Positron (2)
**Architecture**: TransformerConv blocks → JumpingKnowledge → AttentionalAggregation → FC head
**Typical Performance**: >97% exact match accuracy

### 2. GroupSplitter
**Purpose**: Per-hit classification in multi-particle groups
**Output**: Per-hit labels `[is_pion, is_muon, is_mip]`
**Architecture**: TransformerConv blocks → Per-node FC head
**Training**: Only on groups with ≥2 particle types

### 3. PionStopRegressor
**Purpose**: Predict 3D pion stopping position
**Output**: Single 3D coordinate `[x, y, z]`
**Loss**: MSE on Euclidean distance
**Typical Performance**: Sub-millimeter accuracy

### 4. EndpointRegressor
**Purpose**: Predict track start/end points
**Output**: Two 3D coordinates `[2, 3]`
**Status**: Model defined, training in development

### 5. PositronAngleModel
**Purpose**: Predict positron emission angle
**Output**: 2 angle components `[theta, phi]`
**Status**: Model defined, training in development

### 6. GroupAffinityModel
**Purpose**: Predict if two groups should be merged
**Output**: Single affinity score
**Status**: Model defined, no training pipeline yet

## Data Format

Preprocessed data is stored in `.npy` files with structure:

```python
# mainTimeGroups_*.npy
[
    [  # Event 0
        array([[coord, z, stripType, energy, time, pdg_mask, ...]]),  # Group 0
        array([[...]]),  # Group 1
        ...
    ],
    ...
]
```

**PDG Encoding** (bitmask):
- `0b00001` (1): Pion
- `0b00010` (2): Muon
- `0b00100` (4): Positron
- `0b01000` (8): Electron
- `0b10000` (16): Other

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type checking
mypy src/
```

### Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

## Contributing

1. Create a feature branch from `main`
2. Make your changes
3. Run tests and formatting checks
4. Submit a pull request

## Roadmap

### Near-term (by March 2025)
- [ ] Complete endpoint finder and positron angle training pipelines
- [ ] Implement pattern builder for track reconstruction
- [ ] Add GPU cluster training support
- [ ] Integrate bitmask labels from simulation framework
- [ ] Develop specialized models for decay-in-flight and AIF events

### Long-term
- [ ] Multi-detector fusion (ATAR + calorimeter + tracker)
- [ ] Mixture of experts for rare event types
- [ ] Real-time inference optimization
- [ ] End-to-end differentiable reconstruction

## References

- PIONEER Experiment: [https://pioneer.physics.berkeley.edu/](https://pioneer.physics.berkeley.edu/)
- PyTorch Geometric: [https://pytorch-geometric.readthedocs.io/](https://pytorch-geometric.readthedocs.io/)

## License

MIT License (see LICENSE file)

## Contact

For questions or issues, please open an issue on GitHub: [https://github.com/jaca230/pioneerML/issues](https://github.com/jaca230/pioneerML/issues)
