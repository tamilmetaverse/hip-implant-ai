@echo off
REM Quick launcher for Hip Implant AI in VS Code

echo Opening Hip Implant AI in VS Code...

REM Check if VS Code is installed
where code >nul 2>&1
if errorlevel 1 (
    echo [ERROR] VS Code 'code' command not found
    echo.
    echo Please ensure VS Code is installed and added to PATH:
    echo 1. Open VS Code
    echo 2. Press Ctrl+Shift+P
    echo 3. Type "Shell Command: Install 'code' command in PATH"
    echo 4. Run the command
    echo.
    pause
    exit /b 1
)

REM Open in VS Code
code .

echo.
echo VS Code opened!
echo.
echo Quick tips:
echo   - Press F5 to run debug configurations
echo   - Press Ctrl+Shift+P for command palette
echo   - See .vscode\README_VSCODE.md for full guide
echo.
