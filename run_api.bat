@echo off
chcp 65001 >nul
echo.
echo ============================================================
echo   راصد Rased - Starting API Server
echo ============================================================
echo.

cd /d "%~dp0"

echo [1/2] Checking dependencies...
python -c "import fastapi, uvicorn" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    pip install fastapi uvicorn
)

echo [2/2] Starting API Server on port 8000...
echo.
echo API Documentation will be available at:
echo   - Swagger UI: http://localhost:8000/docs
echo   - ReDoc: http://localhost:8000/redoc
echo.
echo Press Ctrl+C to stop the server.
echo.

set PYTHONIOENCODING=utf-8
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
