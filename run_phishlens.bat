@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Set console title
title PhishLens Control Center

REM Change to script directory
cd /d "%~dp0"

echo ================================================================
echo           PHISHLENS - AI Browser Threat Intelligence
echo ================================================================
echo.

REM 1. Check Python virtual environment
if not exist .venv\Scripts\python.exe (
  echo [*] Virtual environment not found. Initializing .venv...
  py -m venv .venv 2>nul
  if errorlevel 1 (
    python -m venv .venv 2>nul
    if errorlevel 1 (
      echo [!] Error: Python not found on PATH. Please install Python 3.10+.
      pause
      exit /b 1
    )
  )
  echo [+] Virtual environment initialized.
)

REM 2. Install / Verify Dependencies
echo [*] Checking and updating dependencies...
".venv\Scripts\python" -m pip install -U pip --quiet
if exist requirements.txt (
  ".venv\Scripts\python" -m pip install -r requirements.txt --quiet
) else if exist backend\requirements.txt (
  ".venv\Scripts\python" -m pip install -r backend\requirements.txt --quiet
)
if errorlevel 1 (
  echo [!] Warning: Some dependencies failed to install. Continuing...
) else (
  echo [+] Dependencies verified.
)

REM 3. Check if Port 8000 is occupied
netstat -ano | findstr /R /C:":8000 " >nul 2>&1
if not errorlevel 1 (
  echo [!] Notice: Port 8000 is already active. Existing backend may be running.
) else (
  echo [*] Starting PhishLens API backend on http://127.0.0.1:8000 ...
  start "PhishLens API Server" ".venv\Scripts\python" backend\app.py
  timeout /t 2 /nobreak >nul
)

REM 4. Open Modern Dashboard
if exist dashboard\index.html (
  echo [*] Launching PhishLens Cyberpunk Threat Dashboard...
  start "" dashboard\index.html
)

echo.
echo ================================================================
echo  [+] PhishLens is UP and RUNNING!
echo  
echo  • API Backend: http://127.0.0.1:8000
echo  • Dashboard:   dashboard\index.html
echo  • Extension:   Load 'extension\' in chrome://extensions
echo ================================================================
echo.
pause
endlocal
