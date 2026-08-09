# AI Voice Changer Dockerfile
# Optimized for real-time inference with NVIDIA GPU acceleration and audio passthrough.
# Note: Host audio passthrough requires running container with `--device /dev/snd` flag.

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Prevent interactive prompts during apt installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install System Audio, C++ Build Tools & Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    portaudio19-dev \
    pulseaudio-utils \
    alsa-utils \
    libasound2-dev \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# Upgrade pip and install PyTorch with CUDA 12.1
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy dependency requirements
COPY requirements.txt . /app/
RUN if [ -f "requirements.txt" ]; then python3 -m pip install --no-cache-dir -r requirements.txt; fi

# Install runtime dependencies
RUN python3 -m pip install --no-cache-dir \
    sounddevice \
    numpy \
    scipy \
    obsws-python \
    websocket-client \
    fastapi \
    uvicorn \
    pydantic \
    librosa \
    soundfile

# Copy application source code
COPY . /app

# Create directory structure for models and user presets
RUN mkdir -p /app/models /app/app/presets/user_presets

# Expose Web UI / API Port
EXPOSE 7860

# Healthcheck endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

# Launch FastAPI / Uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
