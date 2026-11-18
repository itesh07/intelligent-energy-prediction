@echo off
setlocal
cd /d "%~dp0"

echo =====================================================================
echo  WATTWISE - Intelligent Energy Consumption Prediction Setup
echo =====================================================================
echo.

where py >nul 2>nul
if %errorlevel% neq 0 (
  where python >nul 2>nul
  if %errorlevel% neq 0 (
    echo [ERROR] Python launcher or python executable not found.
    echo Please install Python 3.11+ and ensure it is added to your PATH.
    pause
    exit /b 1
  )
)

if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Creating virtual environment (.venv)...
  py -3.11 -m venv .venv 2>nul
  if %errorlevel% neq 0 (
    py -m venv .venv 2>nul
    if %errorlevel% neq 0 (
      python -m venv .venv
      if %errorlevel% neq 0 (
        echo [ERROR] Could not create virtual environment.
        pause
        exit /b 1
      )
    )
  )
) else (
  echo [1/3] Existing virtual environment found in .venv.
)

echo [2/3] Activating virtual environment & installing dependencies...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
  echo [ERROR] Dependency installation failed. Check the error log above.
  pause
  exit /b 1
)

echo [3/3] Training baseline ML models on synthetic smart building benchmark...
python train_model.py
if %errorlevel% neq 0 (
  echo [WARNING] Initial model training encountered an issue. You can train via web UI.
)

echo.
echo =====================================================================
echo  Setup Completed Successfully!
echo  Double-click run_windows.bat to launch the WattWise web application.
echo =====================================================================
pause
