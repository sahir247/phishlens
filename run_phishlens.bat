@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Change to script directory
cd /d "%~dp0"

REM Create venv if it doesn't exist
if not exist .venv\Scripts\python.exe (
  echo Creating Python virtual environment...
  py -m venv .venv
  if errorlevel 1 (
    echo Failed to create virtual environment. Ensure Python is installed and 'py' is on PATH.
    pause
    exit /b 1
  )
)

REM Upgrade pip and install requirements
echo Installing/Updating dependencies...
".venv\Scripts\python" -m pip install -U pip
if exist requirements.txt (
  ".venv\Scripts\python" -m pip install -r requirements.txt
) else (
  ".venv\Scripts\python" -m pip install -r backend\requirements.txt
)
if errorlevel 1 (
  echo Dependency installation failed. See messages above.
  pause
  exit /b 1
)

REM Start Flask backend
echo Starting PhishLens backend on http://127.0.0.1:8000 ...
start "PhishLens Backend" ".venv\Scripts\python" backend\app.py

REM Optional: open dashboard
if exist dashboard\index.html (
  echo Opening dashboard page...
  start "PhishLens Dashboard" dashboard\index.html
)

echo All set. Load the Chrome extension from the 'extension' folder via chrome://extensions
endlocal
