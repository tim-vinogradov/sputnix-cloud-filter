@echo off
chcp 65001 >nul
cd /d "%~dp0"
python --version >nul 2>&1
if errorlevel 1 (
  echo Python 3.9+ is required: https://www.python.org/downloads/
  pause
  exit /b 1
)
python -m pip install -r requirements.txt
if exist demo_run rmdir /s /q demo_run
if exist demo_run_masks rmdir /s /q demo_run_masks
xcopy demo demo_run\ /e /i /q >nul
python cloud_filter.py --input demo_run --masks demo_run_masks --gui
pause
