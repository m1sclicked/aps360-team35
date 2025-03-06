@echo off
setlocal enabledelayedexpansion

REM Set Conda environment name
set ENV_NAME=aps360_gpu

echo Starting Conda GPU Environment Setup...

REM Check if Conda is installed
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Conda is not installed or not in PATH.
    pause
    exit /b 1
)

REM Create a clean Conda environment
echo Creating fresh Conda environment...
call conda create -y -n %ENV_NAME% python=3.9 -c conda-forge
if %errorlevel% neq 0 (
    echo ERROR: Failed to create Conda environment.
    pause
    exit /b 1
)

REM Activate the environment
call conda activate %ENV_NAME%
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate Conda environment.
    pause
    exit /b 1
)

REM Install essential dependencies via Conda
echo Installing essential dependencies...
call conda install -y -c conda-forge numpy=1.23.5 pandas scikit-learn matplotlib seaborn tqdm opencv

REM Install PyTorch with CUDA (avoid pip for PyTorch)
echo Installing PyTorch with CUDA support...
call conda install -y -c pytorch -c nvidia pytorch=2.0.1 torchvision=0.15.2 torchaudio=2.0.2 pytorch-cuda=11.8

REM Install TensorFlow via Conda (more stable than pip)
echo Installing TensorFlow...
call conda install -y -c conda-forge tensorflow=2.9.1

REM Install MediaPipe (ensure protobuf compatibility)
echo Installing MediaPipe...
pip install protobuf==3.19.6 flatbuffers==1.12 mediapipe==0.10.7

echo.
echo Conda environment setup complete!
echo.
pause
