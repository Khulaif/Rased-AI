@echo off
chcp 65001 >nul
echo.
echo ============================================================
echo   راصد Rased - AI Fraud Detection Demo
echo ============================================================
echo.

cd /d "%~dp0"

echo [1/2] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python first.
    pause
    exit /b 1
)

echo [2/2] Running Demo Simulation...
echo.

set PYTHONIOENCODING=utf-8
python demo/simulation.py

echo.
echo ============================================================
echo   Demo Complete!
echo ============================================================
pause
