@echo off
setlocal enabledelayedexpansion

REM Default values
set MODEL_TYPE=bilstm
set USE_ENHANCED=true
set NUM_TRIALS=50
set RESUME_STUDY=false
set SAVE_DIR=results\tuning
set USE_ASL_CITIZEN=true
set NUM_CLASSES=10
set DATA_PATH=data\wlasl_data
set CONFIG_FILE=

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :end_parse
if "%~1"=="--model-type" (
    set MODEL_TYPE=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--enhanced" (
    set USE_ENHANCED=true
    shift
    goto :parse_args
)
if "%~1"=="--standard" (
    set USE_ENHANCED=false
    shift
    goto :parse_args
)
if "%~1"=="--trials" (
    set NUM_TRIALS=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--resume" (
    set RESUME_STUDY=true
    shift
    goto :parse_args
)
if "%~1"=="--save-dir" (
    set SAVE_DIR=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--no-asl-citizen" (
    set USE_ASL_CITIZEN=false
    shift
    goto :parse_args
)
if "%~1"=="--classes" (
    set NUM_CLASSES=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--data-path" (
    set DATA_PATH=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--config" (
    set CONFIG_FILE=%~2
    shift
    shift
    goto :parse_args
)
echo Unknown option: %~1
exit /b 1

:end_parse

REM Create save directory
if not exist "%SAVE_DIR%" mkdir "%SAVE_DIR%"

REM Print configuration
echo ============================================
echo ASL Recognition Hyperparameter Tuning
echo ============================================
echo Model type: %MODEL_TYPE%
echo Enhanced model: %USE_ENHANCED%
echo Number of trials: %NUM_TRIALS%
echo Resume study: %RESUME_STUDY%
echo Save directory: %SAVE_DIR%
echo Number of classes: %NUM_CLASSES%
echo Data path: %DATA_PATH%
echo Use ASL Citizen: %USE_ASL_CITIZEN%
if defined CONFIG_FILE echo Config file: %CONFIG_FILE%
echo.
echo Starting tuning process...
echo ============================================
echo.

REM Create timestamp for logging
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "TIMESTAMP=%dt:~0,8%_%dt:~8,6%"
set "LOG_FILE=%SAVE_DIR%\tuning_log_%TIMESTAMP%.txt"

REM Build command - updated to access the Python file in src directory
set CMD=python -m src.hyperparameter_tuning --model-type %MODEL_TYPE% --trials %NUM_TRIALS% --save-dir %SAVE_DIR% --num-classes %NUM_CLASSES% --data-path %DATA_PATH%

REM Add optional flags
if %USE_ENHANCED%==true (
    set CMD=!CMD! --enhanced
)

if %RESUME_STUDY%==true (
    set CMD=!CMD! --resume-study
)

if %USE_ASL_CITIZEN%==true (
    set CMD=!CMD! --use-asl-citizen
)

if defined CONFIG_FILE (
    set CMD=!CMD! --config %CONFIG_FILE%
)

REM Print and execute command
echo Executing: !CMD!
echo Command saved to log file: %LOG_FILE%
echo.

REM Save command to log file
echo %date% %time% - ASL Recognition Hyperparameter Tuning > "%LOG_FILE%"
echo Command: !CMD! >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

REM Execute command
!CMD!

echo.
echo ============================================
echo Tuning completed. Log saved to: %LOG_FILE%
echo ============================================

REM Check if we need to train different models automatically
if "%1"=="--auto-tune-all" (
    echo Starting automatic tuning of all model types...
    
    REM Save current batch file location
    set "CURRENT_DIR=%~dp0"
    
    REM Train standard BiLSTM
    echo.
    echo ============================================
    echo Starting Standard BiLSTM tuning
    echo ============================================
    call "%CURRENT_DIR%%~nx0" --model-type bilstm --standard --trials %NUM_TRIALS% --save-dir "%SAVE_DIR%\bilstm_standard"
    
    REM Train enhanced BiLSTM
    echo.
    echo ============================================
    echo Starting Enhanced BiLSTM tuning
    echo ============================================
    call "%CURRENT_DIR%%~nx0" --model-type bilstm --enhanced --trials %NUM_TRIALS% --save-dir "%SAVE_DIR%\bilstm_enhanced"
    
    REM Train Transformer
    echo.
    echo ============================================
    echo Starting Transformer tuning
    echo ============================================
    call "%CURRENT_DIR%%~nx0" --model-type transformer --trials %NUM_TRIALS% --save-dir "%SAVE_DIR%\transformer"
    
    REM Train TemporalCNN
    echo.
    echo ============================================
    echo Starting TemporalCNN tuning
    echo ============================================
    call "%CURRENT_DIR%%~nx0" --model-type temporalcnn --trials %NUM_TRIALS% --save-dir "%SAVE_DIR%\temporalcnn"
    
    echo.
    echo ============================================
    echo All model tuning completed!
    echo ============================================
)

endlocal