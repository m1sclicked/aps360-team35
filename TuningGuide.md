# Hyperparameter Tuning for ASL Recognition

This document provides instructions for running extensive hyperparameter tuning for the ASL Recognition model. The tuning system is built using Optuna and supports different model architectures, with a focus on optimizing the enhanced BiLSTM model.

## Files Overview

1. `hyperparameter_tuning.py` - The main Python script for hyperparameter optimization
2. `run_tuning.bat` - A batch script for easily running tuning experiments with different configurations
3. `default_tuning_config.json` - Example configuration file with default parameters

## Getting Started

### Prerequisites

Make sure you have the necessary Python packages installed:

```bash
pip install optuna matplotlib pandas numpy torch
```

### Basic Usage

1. Place the hyperparameter tuning script in your project root:

```bash
# Copy hyperparameter_tuning.py to your project root
```

2. Run the batch script with the desired model type:

```bash
# Run tuning for enhanced BiLSTM model
run_tuning.bat --model-type bilstm --enhanced --trials 50

# Run tuning for transformer model
run_tuning.bat --model-type transformer --trials 50

# Run tuning for temporal CNN model
run_tuning.bat --model-type temporalcnn --trials 50
```

### Automatic Tuning of Multiple Models

To automatically run tuning for all model architectures, use:

```bash
run_tuning.bat --auto-tune-all --trials 30
```

This will sequentially tune:
- Standard BiLSTM
- Enhanced BiLSTM
- Transformer
- Temporal CNN

### Command Line Options

The batch script supports the following options:

| Option | Description | Default |
|--------|-------------|---------|
| `--model-type` | Model type (bilstm, transformer, temporalcnn) | bilstm |
| `--enhanced` | Use enhanced BiLSTM model | N/A |
| `--standard` | Use standard BiLSTM model | N/A |
| `--trials` | Number of optimization trials | 50 |
| `--resume` | Resume previous study | false |
| `--save-dir` | Directory to save results | results\tuning |
| `--no-asl-citizen` | Disable ASL Citizen dataset | N/A |
| `--classes` | Number of classes to use | 10 |
| `--data-path` | Path to WLASL dataset | data\wlasl_data |
| `--config` | Path to custom configuration file | N/A |

## Advanced Configuration

### Custom Configuration Files

You can create a custom configuration file to set specific parameters for your tuning experiments:

```bash
run_tuning.bat --model-type bilstm --enhanced --config my_custom_config.json
```

The configuration file should be in JSON format. See `default_tuning_config.json` for an example.

### Resuming Previous Studies

To resume a previous tuning study:

```bash
run_tuning.bat --model-type bilstm --enhanced --resume
```

This uses a SQLite database to store the study results and can continue from where a previous tuning session left off.

## Hyperparameters Being Tuned

The tuning script optimizes different sets of hyperparameters depending on the model type:

### Common Parameters (All Models)
- Learning rate
- Batch size
- Hidden dimension size
- Number of layers
- Dropout rate
- Weight decay
- Data augmentation parameters
- Early stopping patience

### Enhanced BiLSTM Specific Parameters
- Number of attention heads
- Temporal dropout probability
- Feature dropout probability
- Temporal convolution settings
- Cross-resolution attention
- Gated residual connections

### Transformer Specific Parameters
- Number of attention heads
- Feedforward dimension
- Positional encoding type

### Temporal CNN Specific Parameters
- Number of blocks
- Kernel size
- Residual connections

### Loss Function Parameters
- Focal loss gamma
- Confidence penalty weights
- Temperature scaling

## Results and Analysis

After tuning is complete, the following artifacts are generated in the specified save directory:

- `best_config.json` - The configuration with the best parameters
- `trials.csv` - CSV file with all trial results
- `optimization_history.png` - Plot showing optimization progress
- `param_importances.png` - Plot showing parameter importance
- `parallel_coordinate.png` - Parallel coordinate plot of parameters
- `contour_plot.png` - Contour plot for top parameters

Individual trial results can be found in the trials subdirectory, organized by trial ID and timestamp.

## Best Practices

1. **Start with fewer trials** (15-20) to get a rough idea of which parameters matter
2. **Focus on important parameters** by creating a custom config that narrows search ranges
3. **Use the same random seed** for reproducibility
4. **Monitor system resources** as training models can be memory-intensive

## Troubleshooting

- If you encounter memory issues, try reducing the batch size in your configuration
- For faster tuning, reduce the number of epochs and rely on early stopping
- If models are underfitting, increase the range for hidden dimensions and number of layers
- If models are overfitting, focus on tuning regularization parameters like dropout and weight decay