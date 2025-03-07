@echo off
setlocal enabledelayedexpansion

REM Set Conda environment name
set ENV_NAME=aps360_gpu

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

REM Create a fresh Conda environment with specific configurations
call conda create -y -n %ENV_NAME% python=3.10 pip setuptools wheel

REM Activate the Conda environment
call conda activate %ENV_NAME%

REM Upgrade pip and core packages
python -m pip install --upgrade pip setuptools wheel

REM Install NumPy 1.x first (IMPORTANT: must be installed before TensorFlow)
pip install "numpy<2.0.0"

REM Install TensorFlow with compatible dependencies
pip install tensorflow-cpu

REM Install TensorFlow DirectML plugin
pip install tensorflow-directml-plugin

REM PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

REM Install other dependencies with version constraints
pip install pandas scikit-learn matplotlib seaborn tqdm
pip install opencv-python
pip install kaggle

REM Create a separate Python script for diagnostics with more detailed error handling
echo import sys > gpu_diagnostic.py
echo import platform >> gpu_diagnostic.py
echo print('System Diagnostics:') >> gpu_diagnostic.py
echo print(f'Python: {sys.version}') >> gpu_diagnostic.py
echo print(f'Platform: {platform.platform()}') >> gpu_diagnostic.py
echo try: >> gpu_diagnostic.py
echo     import numpy as np >> gpu_diagnostic.py
echo     print(f'\nNumPy Version: {np.__version__}') >> gpu_diagnostic.py
echo except ImportError as e: >> gpu_diagnostic.py
echo     print(f'NumPy Import Error: {e}') >> gpu_diagnostic.py
echo try: >> gpu_diagnostic.py
echo     import torch >> gpu_diagnostic.py
echo     print('\nPyTorch Information:') >> gpu_diagnostic.py
echo     print(f'Version: {torch.__version__}') >> gpu_diagnostic.py
echo     print(f'CUDA Available: {torch.cuda.is_available()}') >> gpu_diagnostic.py
echo     if torch.cuda.is_available(): >> gpu_diagnostic.py
echo         print(f'CUDA Version: {torch.version.cuda}') >> gpu_diagnostic.py
echo         print(f'GPU Device: {torch.cuda.get_device_name(0)}') >> gpu_diagnostic.py
echo except ImportError as e: >> gpu_diagnostic.py
echo     print(f'PyTorch Import Error: {e}') >> gpu_diagnostic.py
echo try: >> gpu_diagnostic.py
echo     import tensorflow as tf >> gpu_diagnostic.py
echo     print('\nTensorFlow Information:') >> gpu_diagnostic.py
echo     print(f'Version: {tf.__version__}') >> gpu_diagnostic.py
echo     print('Checking DirectML availability...') >> gpu_diagnostic.py
echo     try: >> gpu_diagnostic.py
echo         from tensorflow.python.framework.errors_impl import NotFoundError >> gpu_diagnostic.py
echo         try: >> gpu_diagnostic.py
echo             print(f'Available GPU devices: {tf.config.list_physical_devices("GPU")}') >> gpu_diagnostic.py
echo             print(f'Available DirectML devices: {tf.config.list_physical_devices("DML")}') >> gpu_diagnostic.py
echo             with tf.device('DML:0'): >> gpu_diagnostic.py
echo                 a = tf.constant([[1.0, 2.0], [3.0, 4.0]]) >> gpu_diagnostic.py
echo                 b = tf.constant([[1.0, 1.0], [1.0, 1.0]]) >> gpu_diagnostic.py
echo                 c = tf.matmul(a, b) >> gpu_diagnostic.py
echo             print('DirectML test: Successful matrix multiplication using DirectML!') >> gpu_diagnostic.py
echo         except NotFoundError as e: >> gpu_diagnostic.py
echo             print(f'DirectML device error: {e}') >> gpu_diagnostic.py
echo         except Exception as e: >> gpu_diagnostic.py
echo             print(f'DirectML test error: {e}') >> gpu_diagnostic.py
echo     except ImportError as e: >> gpu_diagnostic.py
echo         print(f'DirectML plugin import error: {e}') >> gpu_diagnostic.py
echo except ImportError as e: >> gpu_diagnostic.py
echo     print(f'TensorFlow Import Error: {e}') >> gpu_diagnostic.py

REM Run the diagnostic script
python gpu_diagnostic.py

echo Conda environment setup complete!
pause