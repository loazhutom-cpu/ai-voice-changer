# AI Voice Changer - Windows PowerShell Setup Script

Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "  AI Voice Changer - Windows Environment Setup" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

# 1. Check Python installation
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[OK] Detected Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Please install Python 3.11 from python.org and select 'Add Python to PATH'." -ForegroundColor Yellow
    Exit 1
}

# 2. Create and Activate Virtual Environment
if (-not (Test-Path "venv")) {
    Write-Host "[INFO] Creating Python virtual environment 'venv'..." -ForegroundColor Yellow
    python -m venv venv
}

Write-Host "[INFO] Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# 3. Check NVIDIA GPU and CUDA
Write-Host "[INFO] Checking GPU / CUDA availability..." -ForegroundColor Yellow
try {
    $nvidiaSmi = nvidia-smi 2>&1
    Write-Host "[OK] NVIDIA GPU detected via nvidia-smi." -ForegroundColor Green
    Write-Host "[INFO] Installing PyTorch with CUDA 12.1 support..." -ForegroundColor Yellow
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
} catch {
    Write-Host "[WARNING] NVIDIA GPU not found or nvidia-smi missing. Falling back to PyTorch CPU build." -ForegroundColor Yellow
    pip install torch torchaudio
}

# 4. Install Python Dependencies
Write-Host "[INFO] Installing application dependencies..." -ForegroundColor Yellow
pip install sounddevice numpy scipy obsws-python websocket-client fastapi uvicorn pydantic librosa soundfile

# 5. Check VB-CABLE Audio Driver
Write-Host "[INFO] Checking for VB-Audio Virtual Cable..." -ForegroundColor Yellow
if (Get-Command winget -ErrorAction SilentlyContinue) {
    try {
        winget list "VB-Audio Virtual Cable"
        Write-Host "[OK] VB-Audio Virtual Cable is installed." -ForegroundColor Green
    } catch {
        Write-Host "[INFO] VB-Cable not detected. Installing via winget..." -ForegroundColor Yellow
        winget install --id VB-Audio.VirtualCable --silent --accept-package-agreements --accept-source-agreements
    }
} else {
    Write-Host "[NOTICE] winget not available. Please verify VB-Cable is installed from https://vb-audio.com/Cable/" -ForegroundColor Yellow
}

# 6. Launch Backend Server
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "[SUCCESS] Setup complete! Starting AI Voice Changer backend..." -ForegroundColor Green
Write-Host "===================================================" -ForegroundColor Cyan
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
