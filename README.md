# APS360 Team 35 - ASL Gesture Recognition

## Overview
This project focuses on **Sign Language Gesture Recognition** using deep learning techniques.

## Project Setup

### Prerequisites
- Python **3.8+**
- **Anaconda** (for environment management)
- **GPU recommended** (but not required)

### Installation
#### 1. Clone the repository
```sh
git clone <repo-url>
cd aps360-team35
```

#### 2. Set up the environment
##### Without GPU:
```sh
setup.bat
conda activate aps360
```
##### With GPU:
```sh
gpu_setup.bat  # May require driver updates
```

#### 3. Run the project
```sh
python main.py
```

## Dataset Preparation

### Model Storage
- **CNN Model:** `results/cnn/YYYYMMDD_HHMMSS/models/cnn_model.pth`
- **SVM Model:** `results/svm/YYYYMMDD_HHMMSS/models/svm_model.pkl`
- **Scaler:** 
  - CNN: `results/cnn/YYYYMMDD_HHMMSS/models/cnn_scaler.pkl`
  - SVM: `results/svm/YYYYMMDD_HHMMSS/models/svm_scaler.pkl`
  
### Training Artifacts
- **CNN Training Plots:** `results/cnn/YYYYMMDD_HHMMSS/plots/cnn_training_progress.png`
- **SVM Parameter Search Plots:** `results/svm/YYYYMMDD_HHMMSS/plots/svm_parameter_search.png`
- **SVM Learning Curves:** `results/svm/YYYYMMDD_HHMMSS/plots/svm_learning_curves.png`

### Metrics and Logs
- **CNN Training Results:** `results/cnn/YYYYMMDD_HHMMSS/logs/cnn_training_results.txt`
- **SVM Training Results:** `results/svm/YYYYMMDD_HHMMSS/logs/svm_training_results.txt`
- **Metrics CSV:** 
  - CNN: `results/cnn/YYYYMMDD_HHMMSS/logs/cnn_metrics.csv`
  - SVM: `results/svm/YYYYMMDD_HHMMSS/logs/svm_metrics.csv`
  
### Summary Files
- **CNN Summary:** `results/cnn/YYYYMMDD_HHMMSS/cnn_summary.json`
- **SVM Summary:** `results/svm/YYYYMMDD_HHMMSS/svm_summary.json`

*Note: YYYYMMDD_HHMMSS represents the timestamp when the model was trained (e.g., 20250306_181739)*

## Project Structure
```
sign-language-gesture-recognition/
│
├── data/
│   └── wlasl_data/
│       ├── WLASL_v0.3.json
│       └── video_files/
│
├── src/
│   ├── __init__.py
│   ├── dataset.py          # Contains GestureDataset class and prepare_dataset/download_wlasl_data functions
│   ├── cnn_model.py        # Contains GestureCNN class for CNN implementation
│   ├── svm_model.py        # Contains GestureSVM class for SVM implementation
│   └── train.py            # Contains train_model and train_svm_model functions
│
├── results/
│   ├── cnn/                # Results specific to CNN model (created by main.py)
│   │   └── YYYYMMDD_HHMMSS/  # Timestamp folders for each run
│   │       ├── models/     # Saved model files
│   │       ├── logs/       # Training logs and metrics
│   │       └── plots/      # Visualizations of training progress
│   │
│   ├── svm/                # Results specific to SVM model (created by main.py)
│   │   └── YYYYMMDD_HHMMSS/  # Timestamp folders for each run
│   │       ├── models/     # Saved model files
│   │       ├── logs/       # Training logs and metrics
│   │       └── plots/      # Visualizations and parameter search results
│   │
│   └── unknown/            # For any other model types (created by main.py)
│
├── requirements.txt
├── README.md
└── main.py                 # Main script with save_results function and main() function
```

---

