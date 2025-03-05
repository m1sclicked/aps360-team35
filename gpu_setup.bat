@echo off
setlocal enabledelayedexpansion

REM Set Conda environment name
set ENV_NAME=aps_360_gpu

REM Verbose error handling
echo Starting Conda GPU Environment Setup...

REM Check if Conda is installed
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Conda is not installed or not in PATH.
    echo Please install Anaconda or Miniconda from https://www.anaconda.com/products/distribution
    pause
    exit /b 1
)

REM Initialize Conda for the current shell with more verbose output
call conda init cmd.exe

REM Create Conda environment with specific configurations
call conda create -y -n %ENV_NAME% python=3.9 pip setuptools wheel

REM Activate the Conda environment
call conda activate %ENV_NAME%

REM Upgrade pip and core packages
python -m pip install --upgrade pip setuptools wheel

REM Uninstall conflicting packages
pip uninstall -y protobuf tensorflow tensorflow-cpu tensorflow-directml-plugin mediapipe tensorboard

REM Install compatible versions of protobuf
pip install 'protobuf>=4.25.3,<5.0.0'

REM Install TensorFlow
pip install tensorflow==2.15.0

REM Try alternative DirectML plugin installation
pip install intel-tensorflow
pip install tensorflow-directml-plugin

REM PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

REM Install other dependencies with version constraints
pip install numpy pandas scikit-learn matplotlib seaborn tqdm opencv-python
pip install 'mediapipe>=0.10.21'
pip install kaggle

REM Create a separate Python script for diagnostics with more detailed error handling
echo import sys > gpu_diagnostic.py
echo import platform >> gpu_diagnostic.py
echo print('System Diagnostics:') >> gpu_diagnostic.py
echo print(f'Python: {sys.version}') >> gpu_diagnostic.py
echo print(f'Platform: {platform.platform()}') >> gpu_diagnostic.py
echo try: >> gpu_diagnostic.py
echo     import torch >> gpu_diagnostic.py
echo     print('\nPyTorch Information:') >> gpu_diagnostic.py
echo     print(f'Version: {torch.__version__}') >> gpu_diagnostic.py
echo     print(f'CUDA Available: {torch.cuda.is_available()}') >> gpu_diagnostic.py
echo     if torch.cuda.is_available(): >> gpu_diagnostic.py
echo         print(f'CUDA Version: {torch.version.cuda}') >> gpu_diagnostic.py
echo         print(f'GPU Device: {torch.cuda.get_device_name(0)}') >> gpu_diagnostic.py
echo except ImportError: >> gpu_diagnostic.py
echo     print('PyTorch not installed') >> gpu_diagnostic.py
echo try: >> gpu_diagnostic.py
echo     import tensorflow as tf >> gpu_diagnostic.py
echo     print('\nTensorFlow Information:') >> gpu_diagnostic.py
echo     print(f'Version: {tf.__version__}') >> gpu_diagnostic.py
echo     try: >> gpu_diagnostic.py
echo         from tensorflow_directml_plugin import load_op_library >> gpu_diagnostic.py
echo         load_op_library() >> gpu_diagnostic.py
echo         print('DirectML Plugin Loaded Successfully') >> gpu_diagnostic.py
echo     except ImportError as e: >> gpu_diagnostic.py
echo         print(f'DirectML Plugin Import Error: {e}') >> gpu_diagnostic.py
echo         print('Detailed DirectML Plugin Installation Check:') >> gpu_diagnostic.py
echo         import pkg_resources >> gpu_diagnostic.py
echo         try: >> gpu_diagnostic.py
echo             print(pkg_resources.get_distribution('tensorflow-directml-plugin')) >> gpu_diagnostic.py
echo         except pkg_resources.DistributionNotFound: >> gpu_diagnostic.py
echo             print('tensorflow-directml-plugin not found in installed packages') >> gpu_diagnostic.py
echo     gpus = tf.config.list_physical_devices('GPU') >> gpu_diagnostic.py
echo     print(f'GPU Devices: {gpus}') >> gpu_diagnostic.py
echo except ImportError as e: >> gpu_diagnostic.py
echo     print(f'TensorFlow Import Error: {e}') >> gpu_diagnostic.py

REM Run the diagnostic script
python gpu_diagnostic.py

echo Conda environment setup complete!
pause