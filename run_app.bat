@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title SafeStep

set "VPY=%~dp0.venv\Scripts\python.exe"

echo.
echo  ============================================================
echo    SafeStep   Fall Detection and Activity Recognition
echo    Daffodil Robo Tech Olympiad
echo  ============================================================
echo.

if not exist "%VPY%" (
    echo  First Launch Detected. Running the One Time Setup.
    echo.
    call "%~dp0setup.bat" --auto || goto :fail
)

"%VPY%" "%~dp0app.py" --check >nul 2>&1
if errorlevel 1 (
    echo  Preparing Models and Data
    "%VPY%" "%~dp0app.py" --prepare || goto :fail
    echo.
)

rem Skip the Streamlit welcome prompt on the first launch
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
    > "%USERPROFILE%\.streamlit\credentials.toml" echo [general]
    >> "%USERPROFILE%\.streamlit\credentials.toml" echo email = ""
)

echo  Starting SafeStep. The Browser Opens Automatically.
echo  Default Address : http://localhost:8501
echo  Keep This Window Open while Presenting. Close It to Stop SafeStep.
echo.

"%VPY%" -m streamlit run "%~dp0app.py" ^
    --server.headless false ^
    --server.runOnSave false ^
    --server.fileWatcherType none ^
    --browser.gatherUsageStats false ^
    --client.toolbarMode minimal ^
    --client.showErrorDetails true ^
    --theme.base light ^
    --theme.primaryColor "#E4572E" ^
    --theme.backgroundColor "#F5F7FB" ^
    --theme.secondaryBackgroundColor "#EAEFF6" ^
    --theme.textColor "#1F2A44"
if errorlevel 1 goto :fail
exit /b 0

:fail
echo.
echo  [Error] SafeStep Could Not Start. Run setup.bat and Read the Messages Above.
echo.
pause
exit /b 1
