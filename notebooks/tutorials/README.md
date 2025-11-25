# PIONEER ML Tutorials

Welcome to the PIONEER ML tutorial series! This collection of Jupyter notebooks will guide you through building, training, and evaluating machine learning models using the PIONEER ML framework and ZenML.

## Tutorial Overview

### [00_zenml_quickstart.ipynb](00_zenml_quickstart.ipynb) - ZenML Quickstart
**Duration**: 15-20 minutes
**Level**: Beginner

Get started with ZenML and the PIONEER ML framework. Learn how to:
- Set up ZenML for quickstart mode (no server required!)
- Run your first ZenML pipeline
- Train a basic GroupClassifier model
- Generate evaluation plots (confusion matrices, ROC curves, etc.)
- Explore the ZenML dashboard

### [01_building_zenml_pipelines.ipynb](01_building_zenml_pipelines.ipynb) - Building ZenML Pipelines
**Duration**: 30-45 minutes
**Level**: Beginner

Master the fundamentals of ZenML pipeline development:
- Understanding pipeline steps and data flow
- Creating modular, reusable pipeline components
- Running and monitoring pipeline execution
- ZenML caching and performance optimization
- Best practices for pipeline organization

### [02_custom_models_training.ipynb](02_custom_models_training.ipynb) - Custom Models and Advanced Training
**Duration**: 45-60 minutes
**Level**: Intermediate

Build custom models and implement advanced training techniques:
- Creating custom Graph Neural Network architectures
- Integrating custom models with ZenML pipelines
- Advanced training configurations (early stopping, learning rate scheduling)
- Hardware acceleration (automatic GPU detection)
- Model evaluation and performance comparison

### [03_hyperparameter_tuning.ipynb](03_hyperparameter_tuning.ipynb) - Hyperparameter Tuning with Optuna
**Duration**: 45-60 minutes
**Level**: Intermediate

Optimize model performance with automated hyperparameter tuning:
- Introduction to Optuna and Bayesian optimization
- Integrating Optuna with ZenML pipelines
- Advanced sampling strategies and early pruning
- Analyzing optimization results and parameter importance
- Best practices for efficient hyperparameter search

### [04_evaluation_plots.ipynb](04_evaluation_plots.ipynb) - Advanced Evaluation and Custom Plots
**Duration**: 45-60 minutes
**Level**: Intermediate

Master advanced model evaluation techniques:
- Comprehensive evaluation metrics beyond accuracy
- Creating custom evaluation plots and visualizations
- Error analysis and misclassification patterns
- Threshold analysis and confidence calibration
- Building evaluation pipelines in ZenML

## Prerequisites

Before starting the tutorials, make sure you have:

1. **Python Environment**: The tutorials assume you're using the `pioneerml` conda environment
2. **PIONEER ML Framework**: All required dependencies should be installed
3. **Basic Python Knowledge**: Familiarity with Python syntax and basic concepts
4. **Machine Learning Basics**: Understanding of training, validation, and evaluation

## Getting Started

1. **Activate your environment**:
   ```bash
   conda activate pioneerml
   ```

2. **Navigate to the tutorials directory**:
   ```bash
   cd notebooks/tutorials
   ```

3. **Start with Tutorial 0**:
   ```bash
   jupyter notebook 00_zenml_quickstart.ipynb
   ```

4. **Follow the tutorials in order** - each builds on concepts from previous ones

## Key Concepts Covered

### ZenML Fundamentals
- Pipeline and step decorators
- Data flow between steps
- Caching and artifact management
- Local vs remote execution

### PIONEER ML Framework
- Graph neural network models
- Custom model architectures
- Data modules and loaders
- Evaluation and plotting utilities

### Advanced ML Techniques
- Custom loss functions and metrics
- Hyperparameter optimization
- Model evaluation strategies
- Performance analysis and debugging

## Hardware Requirements

The tutorials are designed to work on:
- **CPU-only systems**: All tutorials will run, but training may be slower
- **GPU systems**: Automatic GPU detection and utilization
- **Apple Silicon**: MPS acceleration support

## Expected Outputs

Each tutorial generates various outputs:
- **Trained models**: Saved in ZenML artifacts
- **Evaluation plots**: PNG files in the `outputs/` directory
- **Metrics and results**: Displayed in notebook cells and saved as artifacts
- **ZenML runs**: Tracked in the local ZenML store

## Troubleshooting

### Common Issues

1. **ZenML Connection Errors**: The tutorials use in-memory mode - no server setup required
2. **GPU Not Detected**: Check PyTorch CUDA installation with `torch.cuda.is_available()`
3. **Import Errors**: Ensure you're in the correct conda environment
4. **Plot Display Issues**: Some plots may not display in certain environments - they are still saved to files

### Getting Help

- Check the main project README for setup instructions
- Review error messages carefully - they often contain helpful information
- Ensure all dependencies are installed: `pip install -e .`
- Try running individual cells if a tutorial hangs

## Advanced Topics (Future Tutorials)

We're planning additional tutorials covering:
- **Production Deployment**: Serving models in production
- **Experiment Tracking**: Advanced experiment management
- **Data Versioning**: Managing datasets and features
- **Model Monitoring**: Production model performance monitoring
- **A/B Testing**: Running controlled experiments

## Contributing

Found an issue with a tutorial? Want to suggest improvements?

1. Check existing issues in the project repository
2. Create a clear, descriptive issue with steps to reproduce
3. Include your environment details (OS, Python version, etc.)
4. Suggest specific improvements or corrections

## Next Steps

After completing all tutorials, you'll be able to:
- Build complex ML pipelines with ZenML
- Create custom GNN models for graph data
- Optimize hyperparameters efficiently
- Evaluate models comprehensively
- Deploy ML systems with confidence

**Happy learning! 🚀**
