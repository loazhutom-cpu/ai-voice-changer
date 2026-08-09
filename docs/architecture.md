# System Architecture & Technical Specification

This document details the architectural design, component interactions, buffer management algorithms, and technical trade-offs of the Real-Time AI Voice Changer system.

---

## 1. High-Level System Architecture

The application uses an event-driven, decoupled pipeline architecture composed of five main subsystems:

1. **Audio Capture & Input DSP Layer**: Interfaces with host audio drivers, manages hardware ring buffers, and executes pre-inference noise suppression.
2. **Feature Extraction & AI Inference Engine**: Extracts phonetic content vectors and pitch data, invoking neural networks for timbre conversion.
3. **Post-Processing & FX Chain**: Performs gain staging, formant correction, dynamic range compression, and spatial effects.
4. **Virtual Audio Driver & Routing Subsystem**: Feeds transformed audio into virtual cables for downstream consume apps (OBS, Discord, Zoom).
5. **FastAPI Control Plane & Web Interface**: Provides REST/WebSocket API endpoints for parameter adjustment, model hot-swapping, and state monitoring.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             AUDIO INGESTION LAYER                                │
│  ┌──────────────────────┐    ┌──────────────────────┐    ┌────────────────────┐  │
│  │ Physical Microphone  │ -> │ PortAudio Capture    │ -> │ RNNoise Filter     │  │
│  └──────────────────────┘    └──────────────────────┘    └────────────────────┘  │
└─────────────────────────────────────────┬────────────────────────────────────────┘
                                          │ Audio Frames (PCM float32)
                                          v
┌──────────────────────────────────────────────────────────────────────────────────┐
│                            INFERENCE & AI LAYER                                  │
│  ┌──────────────────────┐    ┌──────────────────────┐    ┌────────────────────┐  │
│  │ Pitch Tracker        │    │ HuBERT Feature       │    │ Neural Generator   │  │
│  │ (Harvest/Crepe/RMVPE)│ -> │ Extraction           │ -> │ (RVC / ONNX / TRT) │  │
│  └──────────────────────┘    └──────────────────────┘    └────────────────────┘  │
└─────────────────────────────────────────┬────────────────────────────────────────┘
                                          │ Converted Waveform
                                          v
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          DSP & POST-PROCESSING LAYER                             │
│  ┌──────────────────────┐    ┌──────────────────────┐    ┌────────────────────┐  │
│  │ Formant / Pitch FX   │ -> │ Parametric EQ & Comp │ -> │ Peak Limiter       │  │
│  └──────────────────────┘    └──────────────────────┘    └────────────────────┘  │
└─────────────────────────────────────────┬────────────────────────────────────────┘
                                          │ Processed Audio Stream
                                          v
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           OUTPUT & ROUTING LAYER                                 │
│  ┌──────────────────────┐    ┌──────────────────────┐    ┌────────────────────┐  │
│  │ Lockless Ring Buffer │ -> │ Virtual Driver Out   │ -> │ OBS / Discord      │  │
│  └──────────────────────┘    └──────────────────────┘    └────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Layer-by-Layer Subsystem Breakdown

### 2.1 Audio I/O & DSP Layer (`src/audio/`)
- **Engine**: Built on top of `PyAudio` / `SoundDevice` bindings interfacing with `PortAudio` (Windows WASAPI / DirectSound, macOS CoreAudio, Linux ALSA / PulseAudio).
- **Buffer Architecture**: Utilizes lock-free circular ring buffers (`RingBuffer`) implemented with double-buffered memory allocations to eliminate lock contention between audio callback threads and GPU worker threads.
- **Pre-DSP Pipeline**:
  - **DCA Elimination**: Removes DC offset biases.
  - **Noise Suppression**: RNNoise deep neural network noise gate operating on 10ms frame chunks.
  - **VAD (Voice Activity Detection)**: WebRTC VAD gating to pause neural inference during prolonged silence, saving GPU cycles.

### 2.2 Feature Extraction & ML Inference Engine (`src/inference/`)
- **Feature Extractor**: Uses ContentVec / HuBERT models (`hubert_base.pt`) to convert audio waves into 256-dimensional semantic representations invariant to speaker identity.
- **Pitch Tracking**:
  - **Harvest**: Highly accurate pitch tracking, robust against noise, execution time ~8–12ms.
  - **RMVPE**: Real-time Mel-frequency VPE model optimized for sub-10ms neural pitch estimation.
  - **Crepe**: High accuracy deep-learning pitch estimator (higher compute requirement).
- **Acoustic Model Generator**: RVC (Retrieval-based Voice Conversion) acoustic decoder converts semantic embeddings + target pitch contour into mel-spectrogram or raw waveform using vector quantization index retrieval.
- **Execution Backends**:
  - **PyTorch (eager/compile)**: Default backend supporting FP16 / MPS / CUDA execution.
  - **ONNX Runtime**: Cross-platform acceleration using CUDA Execution Provider or DirectML.
  - **NVIDIA TensorRT**: Compiled engine files (`.engine`) achieving maximum GPU throughput and minimal frame latency.

### 2.3 Post-Processing & FX Chain (`src/audio/dsp.py`)
- **Formant Pitch Shifting**: Adjusts vocal tract resonance independent of fundamental frequency ($F_0$).
- **Parametric Equalizer**: 5-band IIR filter bank (High pass, Low shelf, Peak 1, Peak 2, High shelf) to shape spectral response.
- **Dynamic Compression**: Soft-knee audio compressor preventing volume spikes during loud vocalization.
- **Limiter & Clipping Protection**: Peak brickwall limiter guaranteeing audio stays under 0 dBFS.

### 2.4 Backend API & Control Plane (`src/api/`)
- **FastAPI / Uvicorn Server**: Operates an asynchronous event loop managing application state and model lifecycle.
- **WebSocket Gateway**: Streams live telemetry data (input RMS volume, output RMS, latency breakdown per frame, GPU memory utilization) to clients at 30Hz.
- **IPC Protocol**: Shared memory pointers / IPC queues for passing control parameters (pitch, gain, model switch triggers) to the background audio thread without causing audio stutters.

---

## 3. Data Flow & Latency Budget Analysis

End-to-end latency is the cumulative time between sound entering the microphone capsule and rendered audio emerging from the virtual speaker interface.

```
+---------------------------------------------------------------------------------------+
|                               LATENCY BUDGET (Target: < 45ms)                         |
+-------------------+--------------------+--------------------+-------------------------+
| Stage             | Duration (ms)      | Optimization Strategy                   |
+-------------------+--------------------+--------------------+-------------------------+
| Hardware Mic Capture|  5.3 ms (256 smp) | WASAPI Exclusive / CoreAudio Low Latency|
| Pre-DSP Noise Gate|  2.0 ms            | C++ Native RNNoise bindings             |
| Feature Extraction|  6.5 ms            | HuBERT ONNX Tensor Core execution       |
| Pitch Estimation  |  7.0 ms            | RMVPE PyTorch CUDA / TensorRT FP16      |
| Model Synthesis   | 14.0 ms            | RVC Decoder TensorRT Engine             |
| Post-DSP & EQ     |  1.5 ms            | Vectorized NumPy / SciPy operations     |
| Virtual Driver Out|  5.3 ms (256 smp) | Ring Buffer lockless transfer           |
+-------------------+--------------------+--------------------+-------------------------+
| TOTAL END-TO-END  | 41.6 ms            | Operational in real-time streams        |
+-------------------+--------------------+--------------------+-------------------------+
```

---

## 4. Key Architectural & Design Decisions

### Decision 1: Frame Chunk Size Trade-off (256 vs 512 vs 1024 samples)
- **Context**: Smaller chunk sizes reduce algorithmic buffering delay but increase CPU overhead and GPU launch overhead.
- **Choice**: Default frame size set to **512 samples at 40kHz (~12.8ms)** with configurable support for **256 samples (~6.4ms)** on high-end GPUs.
- **Rationale**: 512 samples strikes the optimal balance between GPU batch processing efficiency and acceptable human perception delay (< 50ms total).

### Decision 2: TensorRT & ONNX Runtime vs Eager PyTorch
- **Context**: PyTorch eager mode incurs Python GIL and CUDA kernel invocation overhead per frame.
- **Choice**: Implemented an automated export pipeline that converts PyTorch models (`.pth`) to **ONNX** and builds native **NVIDIA TensorRT (`.engine`)** files at startup.
- **Rationale**: TensorRT engine execution reduces model synthesis time by **35% to 50%**, enabling real-time operation on mid-tier hardware (e.g., RTX 3060).

### Decision 3: Decoupled Multithreaded / Multiprocess Engine
- **Context**: Web service endpoints or heavy GUI re-renders must never block the real-time audio thread.
- **Choice**: The real-time audio loop executes in a elevated-priority thread with lockless atomic signal flags. Parameter changes from FastAPI are pushed via lock-free queue rings.
- **Rationale**: Guarantees zero audio dropouts or glitches even under high Web UI load or REST request execution.

### Decision 4: Virtual Driver Interface Strategy
- **Context**: Creating custom kernel drivers for Windows/Mac requires code-signing certificates and kernel installation risks.
- **Choice**: Integrate with established, trusted virtual driver protocols (**VB-Audio Cable** on Windows, **BlackHole** on macOS, and **PipeWire / PulseAudio modules** on Linux).
- **Rationale**: Maximizes compatibility, simplifies installer requirements, and leverages stable operating system virtual audio endpoints.
