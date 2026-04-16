@echo off
echo ====================================
echo HIP IMPLANT AI - REAL MODEL DEMO
echo ====================================
echo.
echo Starting GUI with trained model...
echo.

cd /d "%~dp0"
call venv\Scripts\activate.bat
python demo_gui_real.py

pause
