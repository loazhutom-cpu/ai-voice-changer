#!/usr/bin/env bash
# AI Voice Changer - Linux Setup Script (PulseAudio / PipeWire)

set -e

echo "==================================================="
echo "  AI Voice Changer - Linux Environment Setup"
echo "==================================================="

# 1. Check system packages for PulseAudio/PipeWire and PortAudio
echo "[INFO] Installing system dependencies (portaudio, pulseaudio-utils, ffmpeg)..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update -y
    sudo apt-get install -y python3-pip python3-venv portaudio19-dev pulseaudio-utils ffmpeg libasound2-dev
elif command -v dnf &> /dev/null; then
    sudo dnf install -y python3-pip portaudio-devel pulseaudio-utils ffmpeg
elif command -v pacman &> /dev/null; then
    sudo pacman -S --noconfirm python-pip portaudio pulseaudio ffmpeg
fi

# 2. Setup PipeWire / PulseAudio Virtual Null Sink
SINK_NAME="AI_Voice_Changer_Sink"
echo "[INFO] Creating virtual audio sink: ${SINK_NAME}..."

if command -v pactl &> /dev/null; then
    # Check if null-sink already exists
    if pactl list sinks short | grep -q "${SINK_NAME}"; then
        echo "[OK] Virtual sink '${SINK_NAME}' already exists."
    else
        pactl load-module module-null-sink sink_name="${SINK_NAME}" sink_properties=device.description="${SINK_NAME}"
        echo "[OK] Virtual sink '${SINK_NAME}' created successfully."
    fi
else
    echo "[WARNING] pactl command not found. Ensure PipeWire or PulseAudio is active."
fi

# 3. Create Python Virtual Environment
if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtual environment 'venv'..."
    python3 -m venv venv
fi

echo "[INFO] Activating virtual environment..."
source venv/bin/activate

# 4. Check CUDA support
if command -v nvidia-smi &> /dev/null; then
    echo "[OK] NVIDIA GPU detected. Installing CUDA 12.1 PyTorch..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "[NOTICE] NVIDIA GPU not detected. Installing PyTorch CPU build..."
    pip install torch torchaudio
fi

# 5. Install Dependencies
echo "[INFO] Installing Python dependencies..."
pip install --upgrade pip
pip install sounddevice numpy scipy obsws-python websocket-client fastapi uvicorn pydantic librosa soundfile

# 6. Launch Backend Server
echo "==================================================="
echo "[SUCCESS] Linux Setup complete! Starting AI Voice Changer backend..."
echo "==================================================="
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
