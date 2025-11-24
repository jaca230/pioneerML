# PIONEER ML

Machine learning pipeline framework for the PIONEER experiment's AI-based reconstruction system.

## Overview

PIONEER ML provides Graph Neural Network (GNN) models and training pipelines for reconstructing particle physics events from the PIONEER Active Target (ATAR) detector. The system processes time-grouped detector hits through multiple AI models to identify particles (pions, muons, positrons) and extract physics features.

### Key Features

- **DAG-Based Pipeline Framework**: Flexible, composable workflow system
  - Arbitrary processing stages (no prescriptive types)
  - Automatic dependency resolution
  - Shared context for inter-stage communication
  - Easy to extend and customize
- **Standardized Graph Neural Networks**: Transformer-based GNN architectures for various reconstruction tasks
- **Multiple Reconstruction Models**:
  - Time group classification (particle identification)
  - Multi-particle hit splitting
  - Pion stop position regression
  - Track endpoint finding
  - Positron angle prediction
- **Production-Ready**: Complete with tests, documentation, and examples

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
   # All dependencies (core + dev tools)
   pip install -r requirements.txt

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

### Conda environment (recommended)

For an industry-standard, reproducible setup you can create the provided Conda environment:

```bash
conda env create -f env/environment.yml
conda activate pioneerml
```

### uv (fast pip/venv)

If you prefer uv for speedy installs:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # if uv not installed
./env/uv-setup.sh           # core deps in .venv
./env/uv-setup.sh --dev     # core + dev/test deps
source .venv/bin/activate
```

See `env/README.md` for a summary of all environment options.

### Optional Dependencies

- **TensorBoard logging support** (PyTorch Lightning is installed by default):
  ```bash
  pip install tensorboard
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

### Pipeline Framework

Build flexible ML workflows using the DAG-based pipeline system:

```python
from pioneerml.pipelines import Pipeline, Stage, StageConfig, Context, FunctionalStage

# Define stages
def load_data(ctx):
    ctx['dataset'] = load_my_data()

def train_model(ctx):
    model = GroupClassifier(hidden=200, num_blocks=2)
    trained = train(model, ctx['dataset'])
    ctx['model'] = trained

# Create pipeline
pipeline = Pipeline([
    FunctionalStage(
        config=StageConfig(name='load', outputs=['dataset']),
        func=load_data
    ),
    FunctionalStage(
        config=StageConfig(name='train', inputs=['dataset'], outputs=['model']),
        func=train_model
    ),
])

# Run
ctx = pipeline.run()
print(ctx['model'])
```

See [notebooks/examples/pipeline_demo.py](notebooks/examples/pipeline_demo.py) for complete examples.

### Training Models

The repository includes Jupyter notebooks for training each model:

- `notebooks/training/classify_groups.ipynb` - Train the group classifier
- `notebooks/training/group_splitter.ipynb` - Train the hit splitter
- `notebooks/training/pion_stop.ipynb` - Train pion stop regression
- `notebooks/training/endpoint_finder.ipynb` - Track endpoint prediction (in development)
- `notebooks/training/positron_angle.ipynb` - Positron angle prediction (in development)

### Example: Group Classification

```python
from pioneerml.models import GroupClassifier
from pioneerml.data import GraphGroupDataset

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

### Evaluation & Diagnostics

Compute standardized metrics and plots with the CLI once you have predictions and targets saved as `.npy` or `.pt` tensors:

```bash
pioneerml evaluate \
  --predictions outputs/preds.npy \
  --targets outputs/targets.npy \
  --task multilabel \
  --plots multilabel_confusion precision_recall \
  --save-dir outputs/evaluation
```

This produces registered metrics (subset accuracy, precision/recall/F1, AUC/AP) and saves plots such as per-class confusion matrices and PR/ROC curves.

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
├── src/pioneerml/            # Package source
├── tests/                    # Unit tests
├── notebooks/                # Training/evaluation notebooks
├── env/                      # Environment setups (conda/uv)
│   ├── environment.yml
│   ├── uv-setup.sh
│   └── README.md
├── deprecated/legacy_networks/ # Archived implementations
├── requirements.txt          # All dependencies
├── pyproject.toml            # Project configuration
└── README.md                 # This file
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
