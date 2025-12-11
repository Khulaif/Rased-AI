@echo off
chcp 65001 >nul
echo.
echo ============================================================
echo   راصد Rased - Opening Web Interface
echo ============================================================
echo.

cd /d "%~dp0"

echo Opening index.html in your default browser...
start "" "index.html"

echo.
echo Web interface opened successfully!
echo.
pause
