@echo off

REM Check if Conda is installed
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Conda is not installed or not in PATH. Please install Conda first.
    exit /b 1
)

REM Create a new Conda environment named "aps360" with Python 3.9
call conda create -n aps360 python=3.9 -y

REM Upgrade pip using the environment's Python
call conda run -n aps360 python -m pip install --upgrade pip

REM Install dependencies from requirements.txt if the file exists
if exist requirements.txt (
    call conda run -n aps360 python -m pip install -r requirements.txt
) else (
    echo WARNING: requirements.txt not found. Skipping dependency installation.
)

REM Optional: Install Kaggle CLI if not already installed
call conda run -n aps360 python -m pip install kaggle

echo Conda environment setup complete!
pause
