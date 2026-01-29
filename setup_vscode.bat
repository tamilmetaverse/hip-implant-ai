@echo off
REM Setup script for Hip Implant AI in VS Code

echo ====================================
echo Hip Implant AI - VS Code Setup
echo ====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.10+ and try again
    pause
    exit /b 1
)

echo [1/5] Checking Python version...
python --version

echo.
echo [2/5] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo Virtual environment created
) else (
    echo Virtual environment already exists
)

echo.
echo [3/5] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [4/5] Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo [5/5] Verifying installation...
python verify_installation.py

echo.
echo ====================================
echo Setup Complete!
echo ====================================
echo.
echo Next steps:
echo   1. Open VS Code: code .
echo   2. Install recommended extensions when prompted
echo   3. Select Python interpreter: venv\Scripts\python.exe
echo   4. Start coding!
echo.
echo Press F5 in VS Code to run debug configurations
echo See .vscode\README_VSCODE.md for more info
echo.

pause
