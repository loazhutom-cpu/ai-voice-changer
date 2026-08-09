# Real-Time AI Voice Changer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%20%7C%2012.1-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

A low-latency, real-time AI voice conversion application designed for streaming, gaming, content creation, and online communications. The system captures live audio from your microphone, processes it through deep-learning voice conversion models (RVC / SoftVC / Hubert), applies real-time DSP post-processing, and routes the output to virtual audio drivers for seamless integration with **OBS Studio**, **Discord**, **Zoom**, **Twitch**, and other streaming or calling platforms.

---

## Key Features

- **Ultra-Low Latency Pipeline**: End-to-end latency targets of `< 50ms` using optimized buffer management, ONNX Runtime, and TensorRT FP16/INT8 inference execution.
- **Deep Learning Voice Conversion**: SoftVC / ContentVec / Hubert feature extraction paired with retrieval-based voice conversion (RVC) models and pitch-guided conversion.
- **Real-Time Noise Suppression & DSP**: Integrated RNNoise / WebRTC VAD pre-filtering, pitch correction (autotune), formant shifting, parametric EQ, compressor, and reverb.
- **Cross-Platform Virtual Audio Routing**: Native integration with VB-Audio Cable (Windows), BlackHole (macOS), and PipeWire/PulseAudio (Linux).
- **REST & WebSocket API**: Full control via FastAPI backend with real-time streaming WebSockets and low-latency control bindings.
- **Modern Web GUI & System Tray**: Responsive web dashboard for real-time model switching, audio parameter tweaking, gain monitoring, and preset management.
- **Custom Voice Training Tooling**: Integrated tools for dataset preprocessing, pitch extraction, training execution, and index file creation.
- **Safety & Ethical Safeguards**: Built-in audio watermarking and consent verification protocols to ensure responsible usage.

---

## Architecture Overview

The system operates as a modular, multi-stage pipeline designed to minimize audio chunk latency while maintaining high voice fidelity.

```
+------------------+     +-----------------------+     +---------------------------+
|  Microphone Input| --> | Deep Noise Suppression| --> | Feature Extraction        |
|  (PortAudio/PyAudio) | | (RNNoise / WebRTC VAD) |   | (ContentVec / HuBERT)     |
+------------------+     +-----------------------+     +---------------------------+
                                                                     |
                                                                     v
+------------------+     +-----------------------+     +---------------------------+
| Virtual Mic Out  | <-- | DSP Post-Processing   | <-- | AI Voice Conversion Engine|
| (OBS / Discord)  |     | (Reverb, EQ, Comp)    |     | (RVC / ONNX / TensorRT)   |
+------------------+     +-----------------------+     +---------------------------+
```

### Pipeline Stages
1. **Audio Capture**: Captures 16-bit / 32-bit float audio chunks (160–512 samples) from input hardware via PyAudio / SoundDevice.
2. **Noise Suppression & Pre-DSP**: Removes ambient noise, keyboard clicks, and background hum using neural noise filtering.
3. **AI Voice Conversion**: Extracts phonetic representations using ContentVec / HuBERT embeddings, performs pitch tracking (Harvest / Crepe / PM), and converts timbral characteristics via PyTorch / TensorRT inference models.
4. **Post-Processing Effects**: Applies gain staging, formant control, chorus/reverb, and dynamic compression.
5. **Virtual Audio Output**: Feeds processed PCM audio into virtual audio devices for consumption by third-party applications.

---

## Technology Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Language Core** | Python 3.10+ / C++ extensions | Core application logic and low-level audio buffer management |
| **Backend API** | FastAPI, Uvicorn, WebSockets | Control plane, preset management, streaming control |
| **Inference Engines** | PyTorch, ONNX Runtime, TensorRT | Deep learning model inference and CUDA/DirectML acceleration |
| **Audio I/O & DSP** | PyAudio, SoundDevice, Librosa, PyDub, SciPy | Audio stream acquisition, resamplers, DSP filter chains |
| **Virtual Drivers** | VB-Audio Cable (Win), BlackHole (Mac), PipeWire (Linux) | System-wide virtual microphone routing |
| **GUI Framework** | React / TypeScript, TailwindCSS, WebSockets | Web user interface and real-time visualization |
| **Containerization** | Docker, NVIDIA Container Toolkit | GPU-accelerated container deployment |

---

## Project Structure

```
ai-voice-changer/
├── README.md
├── CONTRIBUTING.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── docker-compose.yml
├── Dockerfile
├── config/
│   ├── default_config.yaml
│   └── audio_profiles.yaml
├── docs/
│   ├── architecture.md
│   ├── obs_setup.md
│   ├── voice_training.md
│   ├── api_reference.md
│   └── deployment.md
├── models/
│   ├── hubert/
│   │   └── hubert_base.pt
│   ├── pretrained/
│   └── weights/
├── src/
│   ├── api/
│   │   ├── main.py
│   │   ├── routers/
│   │   └── websocket.py
│   ├── audio/
│   │   ├── capture.py
│   │   ├── dsp.py
│   │   ├── noise_suppression.py
│   │   └── virtual_device.py
│   ├── inference/
│   │   ├── engine.py
│   │   ├── pitch_tracker.py
│   │   └── rvc_model.py
│   ├── training/
│   │   ├── dataset.py
│   │   ├── preprocess.py
│   │   └── train.py
│   └── utils/
│       ├── config.py
│       └── logger.py
└── tests/
    ├── test_audio.py
    ├── test_inference.py
    └── test_latency.py
```

---

## Installation Instructions

### Prerequisites
- **Python**: 3.10 or higher
- **GPU Driver**: NVIDIA CUDA 11.8 or 12.1 compatible driver (Recommended)
- **C++ Build Tools**: MSVC (Windows) or `build-essential` (Linux) / Xcode Tools (macOS)
- **Virtual Audio Driver**:
  - **Windows**: [VB-Audio Virtual Cable](https://vb-audio.com/Cable/)
  - **macOS**: [BlackHole 2ch/16ch](https://github.com/ExistentialAudio/BlackHole)
  - **Linux**: PulseAudio / PipeWire (`sudo apt install pipewire-audio-client-libraries`)

---

### Windows Installation

```powershell
# 1. Clone repository
git clone https://github.com/your-org/ai-voice-changer.git
cd ai-voice-changer

# 2. Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# 3. Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download pretrained feature extraction models
python scripts/download_models.py
```

---

### macOS Installation (Apple Silicon / Intel)

```bash
# 1. Clone repository
git clone https://github.com/your-org/ai-voice-changer.git
cd ai-voice-changer

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies (PyTorch with MPS support)
pip install -r requirements.txt

# 4. Install PortAudio via Homebrew
brew install portaudio ffmpeg

# 5. Download base models
python3 scripts/download_models.py
```

---

### Linux Installation (Ubuntu/Debian)

```bash
# 1. Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-dev build-essential portaudio19-dev ffmpeg libasound2-dev

# 2. Clone repository
git clone https://github.com/your-org/ai-voice-changer.git
cd ai-voice-changer

# 3. Virtual environment setup
python3 -m venv venv
source venv/bin/activate

# 4. Install PyTorch & packages
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 5. Download base models
python3 scripts/download_models.py
```

---

## Usage Guide

### 1. Launching via Command Line Interface (CLI)
```bash
# Run backend service with default configuration
python -m src.api.main --host 127.0.0.1 --port 8000 --config config/default_config.yaml
```

### 2. Quick Interactive Startup Mode
```bash
# Run real-time voice transformer directly in terminal
python src/audio/stream_runner.py --model models/weights/my_voice.pth --input-device 1 --output-device 2
```

### 3. Accessing Web UI
Navigate to `http://localhost:8000` in your web browser to open the control dashboard:
- Select Input Microphone & Output Virtual Audio Cable.
- Choose Voice Model (`.pth` + `.index` file).
- Adjust Pitch Shift (-12 to +12 semitones).
- Toggle Noise Suppression & DSP Effects.
- Monitor Latency Breakdown and Real-time Audio Visualizer.

---

## OBS Setup Guide

1. Open **OBS Studio** -> **Settings** -> **Audio**.
2. Set **Mic/Auxiliary Audio** to **CABLE Output (VB-Audio Virtual Cable)** on Windows or **BlackHole 2ch** on macOS.
3. In AI Voice Changer UI, set **Output Device** to **CABLE Input (VB-Audio Virtual Cable)**.
4. Speak into your physical microphone; audio will route through AI Voice Changer into OBS in real time.
5. *For detailed step-by-step instructions, see [docs/obs_setup.md](docs/obs_setup.md).*

---

## Discord / Zoom Setup Guide

### Discord
- Go to **User Settings** -> **Voice & Video**.
- **Input Device**: Select `CABLE Output (VB-Audio Virtual Cable)` / `BlackHole 2ch`.
- **Output Device**: Select your physical Headphones/Speakers.
- Disable **Automatically determine input sensitivity** and set threshold low.
- Turn off Discord's native **Krisp Noise Suppression** (to prevent double-filtering artifacting).

### Zoom
- Open **Settings** -> **Audio**.
- **Microphone**: Select `CABLE Output (VB-Audio Virtual Cable)` / `BlackHole 2ch`.
- **Background Noise Suppression**: Set to "Low" or "Off".

---

## Custom Voice Training Guide

Train your own high-fidelity voice model with as little as 10–30 minutes of clean speech.

```bash
# 1. Preprocess audio dataset (slicing, normalization, noise removal)
python src/training/preprocess.py --dataset-dir ./data/my_target_voice/ --output-dir ./logs/my_voice/

# 2. Extract pitch features (Harvest/Crepe) and HuBERT embeddings
python src/training/extract_features.py --config config/train_config.yaml

# 3. Train voice conversion model
python src/training/train.py --model-name my_voice --epochs 200 --batch-size 8

# 4. Export ONNX/PyTorch model and feature index
python src/training/export.py --model-name my_voice --output-dir ./models/weights/
```

*For complete dataset preparation, hyperparameter tuning, and export guides, read [docs/voice_training.md](docs/voice_training.md).*

---

## Configuration Reference

Configurations are managed in `config/default_config.yaml`:

```yaml
audio:
  sample_rate: 40000        # Model sample rate (40kHz or 48kHz)
  block_time: 0.04          # Processing frame size in seconds (40ms)
  extra_convert_size: 0.02  # Overlap buffer size to prevent clicks
  input_device_index: 0     # Hardware Microphone
  output_device_index: 1    # Virtual Audio Cable

inference:
  device: "cuda:0"          # Compute device: cuda:0, mps, or cpu
  precision: "fp16"         # fp32, fp16, int8
  pitch_algorithm: "harvest"# harvest, pm, crepe, rmvpe
  index_rate: 0.6           # Feature retrieval ratio (0.0 to 1.0)

dsp:
  noise_suppression: true
  gate_threshold_db: -45.0
  pitch_shift_semitones: 0
  formant_shift: 1.0
  reverb_mix: 0.05
```

---

## Performance & Latency Targets

| Target Metric | Benchmark Target | Hardware Configuration |
| :--- | :--- | :--- |
| **Inference Time** | 12 - 18 ms | NVIDIA RTX 3060 / 4070 (FP16 / TensorRT) |
| **Audio I/O Buffer** | 10 - 20 ms | PortAudio Block Size 512 @ 48kHz |
| **DSP & Resampling** | 3 - 5 ms | C++ / SciPy Native Filters |
| **End-to-End Latency** | **35 - 48 ms** | Total Real-Time Delay Target |

---

## GPU Requirements

| Requirement Level | Minimum Specs | Recommended Specs | Ultra / Low Latency |
| :--- | :--- | :--- | :--- |
| **GPU** | NVIDIA GTX 1060 (6GB) / Apple M1 | NVIDIA RTX 3060 (12GB) | NVIDIA RTX 4070 / 4080 (16GB) |
| **VRAM** | 4 GB VRAM | 8 GB VRAM | 12+ GB VRAM |
| **CUDA Version** | CUDA 11.8 | CUDA 12.1 | CUDA 12.1 + TensorRT 8.6+ |
| **CPU Fallback** | Supported (High Latency ~200ms) | N/A | N/A |

---

## Safety & Ethics Section

This software provides powerful voice conversion technology. Users must adhere to strict ethical and legal principles:

1. **Consent-Based Cloning**: Never clone or synthesize a person's voice without their explicit, documented authorization.
2. **Synthetic Voice Disclosure**: Always disclose the use of AI voice synthesis when streaming, creating content, or engaging in public communications.
3. **No Deception / Harassment**: Using AI voice changing to deceive, impersonate government/financial/legal figures, perform fraud, or harass individuals is strictly prohibited and illegal in many jurisdictions.
4. **Watermarking**: The application embeds imperceptible acoustic watermarks in generated audio streams to allow downstream origin detection.

---

## Development Roadmap

- [x] **Phase 1: Core Engine & Low-Latency Pipeline**
  - Implement low-latency audio capture and virtual driver routing.
  - Integrate HuBERT / ContentVec feature extractor and RVC pitch tracker.
  - Build basic FastAPI REST control service.

- [ ] **Phase 2: GUI, DSP Enhancement & TensorRT Acceleration**
  - Launch React-based real-time dashboard and visualizer.
  - Add native TensorRT FP16/INT8 compilation pipelines for 30%+ latency reduction.
  - Implement full 5-band EQ, compressor, and spatial reverb DSP suite.

- [ ] **Phase 3: Multi-Voice & Plugin Ecosystem**
  - Support multi-model hot-swapping without audio dropping.
  - VST3 / AU plugin wrapper support for direct DAW integration.
  - Multi-speaker crossfading and real-time accent modification.

- [ ] **Phase 4: Cloud Engine & Embedded Deployment**
  - Standalone C++ engine runtime with zero Python dependency for embedded hardware.
  - Distributed GPU worker orchestration for high-density voice servers.

---

## Contributing

We welcome contributions from the community! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on code style, branch workflows, and PR submissions.

---

## License

This project is released under the terms of the [MIT License](LICENSE).
