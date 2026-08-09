# Deployment & Production Setup Guide

This guide covers production deployment strategies for **Real-Time AI Voice Changer**, including native OS setup, Docker containerization with NVIDIA GPU passthrough, GPU hardware configuration, and driver troubleshooting.

---

## 1. Native System Setup

### 1.1 Windows Native Deployment

#### Requirements
- Windows 10/11 64-bit
- NVIDIA GPU with Driver 525+ (for CUDA 12.x support)
- Visual C++ Redistributable 2019/2022
- [VB-Audio Virtual Cable Driver](https://vb-audio.com/Cable/)

#### Installation Commands
```powershell
# 1. Clone repository
git clone https://github.com/your-org/ai-voice-changer.git
cd ai-voice-changer

# 2. Create isolated Python environment
python -m venv venv
.\venv\Scripts\activate

# 3. Install CUDA-enabled PyTorch
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install audio stack & dependencies
pip install -r requirements.txt

# 5. Launch Service
python -m src.api.main --host 0.0.0.0 --port 8000
```

---

### 1.2 macOS Native Deployment (Apple Silicon M1/M2/M3 & Intel)

#### Requirements
- macOS 12 Monterey or higher
- [BlackHole 2ch Driver](https://github.com/ExistentialAudio/BlackHole)
- Homebrew package manager

#### Installation Commands
```bash
# 1. Install system audio library and FFmpeg
brew install portaudio ffmpeg blackhole-2ch

# 2. Setup Python environment
python3 -m venv venv
source venv/bin/activate

# 3. Install PyTorch with Metal Performance Shaders (MPS) support
pip install -r requirements.txt

# 4. Run application
python3 -m src.api.main --host 127.0.0.1 --port 8000
```

---

### 1.3 Linux Native Deployment (Ubuntu 22.04 LTS / PipeWire / PulseAudio)

#### Requirements
- Ubuntu 22.04 LTS or Debian 12
- NVIDIA Proprietary Driver 535+
- PipeWire or PulseAudio sound server

#### Audio Loopback Driver Creation
```bash
# Create PipeWire virtual source
pactl load-module module-null-sink sink_name=AIVoiceSink sink_properties=device.description="AI_Voice_Virtual_Mic"
pactl load-module module-remap-source master=AIVoiceSink.monitor source_name=AIVoiceMic source_properties=device.description="AI_Voice_Mic"
```

#### Application Setup
```bash
# Install audio build tools
sudo apt-get update && sudo apt-get install -y \
    python3-dev \
    build-essential \
    portaudio19-dev \
    libasound2-dev \
    ffmpeg

# Virtualenv setup
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Run Service
python3 -m src.api.main --host 0.0.0.0 --port 8000
```

---

## 2. Docker Deployment with NVIDIA GPU Passthrough

Docker deployment is ideal for server hosting, remote streaming setups, or headless GPU instances.

### Prerequisites
1. Docker Engine 24.0+
2. [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### `Dockerfile`
```dockerfile
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Avoid tzdata interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    portaudio19-dev \
    ffmpeg \
    libasound2-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CUDA 12.1
RUN pip3 install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy requirement manifest and install
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

EXPOSE 8000

CMD ["python3", "-m", "src.api.main", "--host", "0.0.0.0", "--port", "8000"]
```

---

### `docker-compose.yml`
```yaml
version: "3.8"

services:
  ai-voice-changer:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: ai_voice_changer
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models
      - ./config:/app/config
    devices:
      - "/dev/snd:/dev/snd"  # Sound card passthrough (ALSA)
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

#### Launch Container
```bash
docker compose up -d --build
```

---

## 3. GPU Hardware Configuration & Acceleration Options

### 3.1 NVIDIA CUDA & TensorRT (Recommended)
- **Primary Driver**: CUDA 12.1
- **Optimization Strategy**: Convert `.pth` models to `.engine` format using TensorRT 8.6+ for INT8/FP16 execution.
- **Latency Gain**: ~40% reduction in inference time compared to eager PyTorch.

```bash
# Enable PyTorch CUDNN Benchmarking in config
export CUDNN_BENCHMARK=1
```

### 3.2 AMD GPU DirectML Fallback
For AMD Radeon GPUs on Windows, use ONNX Runtime with DirectML Execution Provider:

```bash
pip install onnxruntime-directml
```

In `config/default_config.yaml`:
```yaml
inference:
  device: "directml:0"
  backend: "onnx"
```

### 3.3 Apple Silicon MPS Acceleration
For Apple M1/M2/M3 chips:
```yaml
inference:
  device: "mps"
  backend: "pytorch"
  precision: "fp32"
```

---

## 4. Troubleshooting Deployment Issues

### Issue 1: CUDA Out of Memory (OOM) Errors
- **Symptom**: `RuntimeError: CUDA out of memory` during model loading or stream launch.
- **Fix**:
  1. Switch pitch algorithm from `crepe` to `rmvpe` or `harvest`.
  2. Lower audio batch processing frame size from `1024` to `512` samples.
  3. Ensure no other VRAM-heavy processes (games with ultra textures) exhaust GPU memory.

### Issue 2: Docker Sound Card Permission Denied
- **Symptom**: `ALSA lib pcm.c:2664:(snd_pcm_open_conf) Cannot open shared library` inside Docker.
- **Fix**:
  Grant ALSA sound group access to Docker:
  ```bash
  sudo usermod -aG audio $USER
  docker run --device /dev/snd --group-add audio ...
  ```

### Issue 3: High Latency (> 100ms) on Laptop GPUs
- **Symptom**: Processed audio is delayed by over 100ms when running on laptops.
- **Fix**:
  1. Ensure Windows Power Plan is set to **High Performance**.
  2. Open NVIDIA Control Panel -> **Manage 3D Settings** -> Set Preferred Graphics Processor to **High-Performance NVIDIA Processor**.
  3. Disable GPU Power Throttling / Battery Saver mode.
