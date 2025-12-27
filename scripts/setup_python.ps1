# PowerShell setup script for Python environment
# Author: s Bostan
# Created on: Nov, 2025
# Windows version

Write-Host "Setting up Python environment for AdaptiveMultimodalRAG..." -ForegroundColor Green

# Check if virtual environment exists
if (Test-Path "venv") {
    Write-Host "Virtual environment already exists." -ForegroundColor Yellow
} else {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& ".\venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# Install requirements
Write-Host "Installing requirements..." -ForegroundColor Cyan
# Try using the venv's pip first, if that fails due to SSL issues, use system pip
try {
    pip install -r requirements.txt
} catch {
    Write-Host "Encountered SSL/proxy issues with venv pip. Trying alternative method..." -ForegroundColor Yellow
    # Alternative: Use system pip to install into venv (workaround for SSL issues)
    $pythonPath = (Get-Command python).Source
    $pipPath = Join-Path (Split-Path $pythonPath) "Scripts\pip.exe"
    if (Test-Path $pipPath) {
        & $pipPath install --target=venv\Lib\site-packages -r requirements.txt
    } else {
        Write-Host "Error: Could not find pip. Please install requirements manually." -ForegroundColor Red
        exit 1
    }
}

Write-Host "Python environment setup complete!" -ForegroundColor Green
Write-Host "To activate the environment, run: .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow

