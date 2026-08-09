#!/usr/bin/env bash
# AI Voice Changer - macOS Setup Script

set -e

echo "==================================================="
echo "  AI Voice Changer - macOS Environment Setup"
echo "==================================================="

# 1. Check Homebrew
if ! command -v brew &> /dev/null; then
    echo "[ERROR] Homebrew is not installed. Please install Homebrew from https://brew.sh"
    exit 1
fi

# 2. Install BlackHole Virtual Audio Driver
echo "[INFO] Checking BlackHole virtual audio driver..."
if brew list --cask blackhole-2ch &> /dev/null || brew list blackhole-2ch &> /dev/null; then
    echo "[OK] BlackHole 2ch audio driver is already installed."
else
    echo "[INFO] Installing BlackHole 2ch via Homebrew..."
    brew install blackhole-2ch
fi

# 3. Check Python 3.11+
if ! command -v python3 &> /dev/null; then
    echo "[INFO] Installing Python 3.11 via Homebrew..."
    brew install python@3.11
fi

# 4. Create Virtual Environment
if [ ! -d "venv" ]; then
    echo "[INFO] Creating Python virtual environment 'venv'..."
    python3 -m venv venv
fi

echo "[INFO] Activating virtual environment..."
source venv/bin/activate

# 5. Install PyTorch & Dependencies
echo "[INFO] Installing PyTorch and Audio Dependencies (MPS / Metal support for macOS)..."
pip install --upgrade pip
pip install torch torchaudio
pip install sounddevice numpy scipy obsws-python websocket-client fastapi uvicorn pydantic librosa soundfile

# 6. Launch Backend Server
echo "==================================================="
echo "[SUCCESS] macOS Setup complete! Starting AI Voice Changer backend..."
echo "==================================================="
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
