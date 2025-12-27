@echo off
REM Batch setup script for Python environment
REM Author: s Bostan
REM Created on: Nov, 2025
REM Windows version

echo Setting up Python environment for AdaptiveMultimodalRAG...

REM Check if virtual environment exists
if exist "venv" (
    echo Virtual environment already exists.
) else (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing requirements...
REM Try using the venv's pip first, if that fails due to SSL issues, use system pip
pip install -r requirements.txt
if errorlevel 1 (
    echo Encountered SSL/proxy issues with venv pip. Trying alternative method...
    REM Alternative: Use system pip to install into venv (workaround for SSL issues)
    where python > temp_python_path.txt
    for /f "tokens=*" %%i in (temp_python_path.txt) do set PYTHON_PATH=%%i
    del temp_python_path.txt
    for %%i in ("%PYTHON_PATH%") do set PIP_PATH=%%~dpiScripts\pip.exe
    if exist "%PIP_PATH%" (
        "%PIP_PATH%" install --target=venv\Lib\site-packages -r requirements.txt
    ) else (
        echo Error: Could not find pip. Please install requirements manually.
        exit /b 1
    )
)

echo Python environment setup complete!
echo To activate the environment, run: venv\Scripts\activate.bat

pause

