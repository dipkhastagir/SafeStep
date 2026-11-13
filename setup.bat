@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title SafeStep Setup

set "AUTO=0"
if /i "%~1"=="--auto" set "AUTO=1"
set "VENV=%~dp0.venv"
set "VPY=%VENV%\Scripts\python.exe"
set "PACKAGES=streamlit>=1.50 plotly>=5.24 pyarrow>=15 numpy==2.4.6 pandas==2.2.3 scipy==1.16.3 scikit-learn==1.6.1 joblib==1.6.0 lightgbm==4.6.0"

echo.
echo  ============================================================
echo    SafeStep   Fall Detection and Activity Recognition
echo    Environment Setup
echo  ============================================================
echo    Project Folder : %~dp0
echo.

rem ---------------------------------------------------------------- Step 1
echo  [1/4] Locating Python 3.11, 3.12 or 3.13
set "PY="
for %%V in (3.13 3.12 3.11) do (
    if not defined PY py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
)
if not defined PY (
    python -c "import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY goto :no_python
for /f "delims=" %%I in ('%PY% -c "import sys; print(sys.version.split()[0])"') do set "PYVER=%%I"
echo        Found Python %PYVER%
echo.

rem ---------------------------------------------------------------- Step 2
echo  [2/4] Preparing the Virtual Environment
if exist "%VPY%" (
    "%VPY%" -c "import sys" >nul 2>&1 || (
        echo        Existing Environment Is Damaged, Rebuilding It
        rmdir /s /q "%VENV%"
    )
)
if not exist "%VPY%" (
    %PY% -m venv "%VENV%" || goto :venv_failed
    echo        Created .venv
) else (
    echo        Reusing .venv
)
echo.

rem ---------------------------------------------------------------- Step 3
echo  [3/4] Installing Libraries  ^(First Run Takes a Few Minutes^)
"%VPY%" -m pip install --upgrade pip --disable-pip-version-check -q
"%VPY%" -m pip install %PACKAGES% --disable-pip-version-check --progress-bar on || goto :pip_failed
echo        Libraries Ready
echo.

rem ---------------------------------------------------------------- Step 4
echo  [4/4] Preparing Models and Data
"%VPY%" "%~dp0app.py" --prepare || goto :bundle_failed
echo.

echo  ============================================================
echo    Setup Complete. Start SafeStep with run_app.bat
echo  ============================================================
echo.
if "%AUTO%"=="0" pause
exit /b 0

:no_python
echo.
echo  [Error] Python 3.11, 3.12 or 3.13 Not Found.
echo          Install Python 3.13 from https://www.python.org/downloads/
echo          and Tick "Add python.exe to PATH" during Installation.
goto :fail

:venv_failed
echo.
echo  [Error] Virtual Environment Creation Failed.
goto :fail

:pip_failed
echo.
echo  [Error] Library Installation Failed. Check the Internet Connection and Run setup.bat Again.
goto :fail

:bundle_failed
echo.
echo  [Error] Models and Data Not Ready. Place SafeStep_bundle.zip or the app_bundle Folder in:
echo          %~dp0SafeStep
goto :fail

:fail
echo.
if "%AUTO%"=="0" pause
exit /b 1
