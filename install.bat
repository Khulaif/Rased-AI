@echo off
chcp 65001 >nul
echo.
echo ============================================================
echo   راصد Rased - Install Dependencies
echo ============================================================
echo.

cd /d "%~dp0"

echo Installing required Python packages...
echo.

pip install -r requirements.txt

echo.
echo ============================================================
echo   Installation Complete!
echo ============================================================
echo.
echo You can now run:
echo   - run_demo.bat     : Run the demo simulation
echo   - run_api.bat      : Start the API server
echo   - open_web.bat     : Open the web interface
echo.
pause
