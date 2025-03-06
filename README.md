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
- **Trained Model:** `results/models/gesture_model.pth`
- **Scaler:** `results/models/gesture_scaler.npy`
- **Training Plots:** `results/plots/training_progress.png`
- **Training Logs:** `results/logs/training_results.txt`

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

