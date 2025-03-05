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
cd sign-language-gesture-recognition
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
│   ├── dataset.py
│   ├── model.py
│   └── train.py
│
├── results/
│   ├── models/
│   ├── logs/
│   └── plots/
│
├── requirements.txt
├── README.md
└── main.py
```

---
Feel free to update this README as necessary!

