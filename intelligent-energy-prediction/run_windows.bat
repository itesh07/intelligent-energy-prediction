@echo off
setlocal
cd /d "%~dp0"

echo =====================================================================
echo  WATTWISE - SMART ENERGY AI APPLICATION LAUNCHER
echo  Intelligent Energy Consumption Prediction System for Smart Buildings
echo =====================================================================
echo.

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Virtual environment not found. Please run setup_windows.bat first.
  pause
  exit /b 1
)

call .venv\Scripts\activate.bat
echo Starting Flask web application...
echo Web Dashboard available at: http://127.0.0.1:5000
echo Press Ctrl+C in this console to stop the server.
echo.

python app.py
pause
