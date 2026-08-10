# Audio Pipeline Documentation

Complete technical reference for the real-time AI voice conversion pipeline.

---

## Overview

The audio pipeline is a multi-stage real-time system that captures microphone input, applies noise suppression, gain control, RVC voice conversion, audio effects, dry/wet mixing, and routes the processed output to a virtual audio device for consumption by OBS, Discord, Zoom, VRChat, and other applications.

```
┌─────────────┐    ┌──────────────┐    ┌───────────┐    ┌──────────────┐    ┌───────────┐    ┌──────────┐    ┌─────────────┐
│  Microphone │──> │ Noise Suppress │──> │ Gain Ctrl │──> │ RVC Conversion│──> │  Effects  │──> │ Dry/Wet  │──> │ Virtual Mic │
│  (sounddevice) │  │ (noisereduce)  │    │ (AGC/Manual) │  │ (PyTorch)     │    │ (Pedalboard) │   │ Mix      │    │ (AudioRouter) │
└─────────────┘    └──────────────┘    └───────────┘    └──────────────┘    └───────────┘    └──────────┘    └─────────────┘
                                              │
                                              v
                                        ┌──────────┐
                                        │ Recording│ (WAV 16-bit PCM)
                                        └──────────┘
```

---

## Module Structure

```
ai-engine/
├── inference/
│   ├── pipeline.py            # AudioPipeline orchestrator (805 lines)
│   ├── audio_buffer.py         # CircularAudioBuffer (thread-safe ring buffer)
│   ├── pitch_extractor.py      # PitchExtractor (F0 estimation + MIDI conversion)
│   ├── feature_extractor.py    # FeatureExtractor (HuBERT/ContentVec embeddings)
│   └── gain_control.py         # GainController (AGC, manual, noise gate, bypass)
├── rvc/
│   ├── rvc_engine.py           # RVCEngine (model loading + voice conversion)
│   └── rvc_inference.py        # RealtimeRVCInference (streaming worker thread)
├── models/
│   └── voice_models.py         # VoiceModelRegistry (preset management)
└── models/presets/             # Voice model .pth files

backend/
├── main.py                     # FastAPI server (REST + WebSocket)
├── audio_effects.py            # AudioEffectsChain (Pedalboard DSP)
└── noise_suppression.py        # NoiseSuppressor (noisereduce)

configs/
└── default.yaml                # Pipeline configuration

app/routing/
└── audio_router.py             # AudioRouter (virtual mic output routing)
```

---

## Component Reference

### 1. AudioPipeline (`ai-engine/inference/pipeline.py`)

The top-level orchestrator that ties all components together. Manages the pipeline lifecycle, configuration, event callbacks, recording, routing, and diagnostics.

#### Lifecycle

```python
from ai_engine.inference.pipeline import AudioPipeline

# Basic usage
pipeline = AudioPipeline()
pipeline.start()
# ... pipeline is running ...
pipeline.stop()

# Context manager (auto start/stop)
with AudioPipeline(config_path="configs/default.yaml") as pipeline:
    pipeline.set_voice_preset("anime_heroine")
    pipeline.set_pitch_shift(-5.0)
    # ... do stuff ...
    # pipeline.stop() called automatically on exit
```

#### Pipeline States

| State | Description |
|---|---|
| `IDLE` | Pipeline initialized but not started |
| `STARTING` | Transition: starting inference worker and streams |
| `RUNNING` | Pipeline is actively processing audio |
| `STOPPING` | Transition: stopping worker and closing streams |
| `STOPPED` | Pipeline stopped, resources released |
| `ERROR` | Pipeline encountered an error |

#### Key Methods

| Method | Description |
|---|---|
| `start()` | Start the real-time pipeline (inference worker + audio router + monitor) |
| `stop()` | Stop gracefully (releases streams, router, recording) |
| `restart()` | Stop + start |
| `set_voice_preset(preset_id)` | Switch active voice model |
| `set_pitch_shift(semitones)` | Set pitch shift (-24 to +24 semitones) |
| `set_dry_wet(value)` | Set dry/wet mix (0.0=dry/original, 1.0=wet/converted) |
| `set_bypass(enabled)` | Enable bypass mode (passthrough without conversion) |
| `set_ptt_enabled(enabled)` | Enable/disable push-to-talk mode |
| `set_ptt_active(active)` | Set PTT transmit state (True=talking, False=muted) |
| `set_output_target(app, device)` | Switch virtual mic output target (obs/discord/zoom/vrchat) |
| `start_recording(path)` | Start recording output to WAV file |
| `stop_recording()` | Stop recording and close file |
| `update_settings(dict)` | Batch update multiple settings at once |
| `get_status()` | Get comprehensive pipeline status snapshot |
| `get_latency()` | Get latency breakdown (processing_ms, buffer_ms, total_ms) |
| `get_audio_levels()` | Get input/output RMS and dB levels |
| `get_buffer_health()` | Get buffer underrun/overrun counts |
| `get_device_info()` | Get GPU/device diagnostics |
| `on(event, callback)` | Register an event callback |
| `off(event, callback)` | Unregister an event callback |

#### Event System

```python
# Register callbacks
pipeline.on("state_change", lambda data: print(f"State: {data}"))
pipeline.on("error", lambda data: print(f"Error: {data}"))
pipeline.on("level_update", lambda data: update_meters(data))
pipeline.on("latency_update", lambda data: update_latency_display(data))

# Unregister
pipeline.off("level_update", my_callback)
```

**Available events:**
- `state_change` — Emitted when pipeline state or preset changes
- `error` — Emitted on any pipeline error (includes context field)
- `level_update` — Emitted at ~2 FPS with input/output RMS levels
- `latency_update` — Emitted at ~2 FPS with processing/buffer/total latency

---

### 2. PitchExtractor (`ai-engine/inference/pitch_extractor.py`)

Extracts fundamental frequency (F0) from audio for pitch-guided voice conversion.

| Method | Description |
|---|---|
| `extract_f0(audio, sr)` | Extract F0 contour in Hz (0.0 = unvoiced) |
| `hz_to_midi(f0_hz)` | Convert Hz array to MIDI note numbers |
| `apply_pitch_shift(f0, semitones)` | Shift F0 contour by N semitones |
| `resample_contour(contour, length)` | Resample contour to target frame count |

**Methods:**
- `autocorrelation` — Fast FFT-based autocorrelation (default, CPU-friendly)
- `pyin` — High-accuracy via librosa.pyin (requires librosa)

```python
from ai_engine.inference.pitch_extractor import PitchExtractor

extractor = PitchExtractor(method="autocorrelation", fmin=65.0, fmax=1000.0)
f0 = extractor.extract_f0(audio_chunk, sample_rate=48000)
midi = extractor.hz_to_midi(f0)
shifted = extractor.apply_pitch_shift(f0, semitones=-5.0)
```

---

### 3. FeatureExtractor (`ai-engine/inference/feature_extractor.py`)

Extracts content embeddings from audio for RVC voice conversion. Uses a lightweight HuBERT/ContentVec-style encoder when the full HuBERT model is unavailable.

| Feature | Details |
|---|---|
| Default encoder | LightweightContentEncoder (stacked Conv1d + GroupNorm + projection) |
| Full model | Tries loading HuBERT via torch.hub (requires fairseq) |
| Fallback | NumPy spectral features when PyTorch unavailable |
| Output | (frames, 256) float32 embeddings |
| Sample rate | Auto-resamples to 16kHz for HuBERT compatibility |

```python
from ai_engine.inference.feature_extractor import FeatureExtractor

extractor = FeatureExtractor(feature_dim=256)
features = extractor.extract(audio_chunk, sample_rate=48000)
# features.shape == (num_frames, 256)
```

---

### 4. GainController (`ai-engine/inference/gain_control.py`)

Real-time gain staging with automatic gain control, manual gain, and noise gate.

| Mode | Description |
|---|---|
| `manual` | Fixed dB gain applied to all audio |
| `agc` | Automatic gain control targeting a specific RMS level |
| `bypass` | No gain adjustment (passthrough) |

**AGC parameters:**
- `agc_target_rms` — Target output RMS level (default: 0.15)
- `agc_attack_ms` — Time to increase gain (default: 20ms)
- `agc_release_ms` — Time to decrease gain (default: 300ms)
- `max_gain_db` / `min_gain_db` — Gain clamping range (±30dB)

```python
from ai_engine.inference.gain_control import GainController

gc = GainController(mode="agc", agc_target_rms=0.15)
processed = gc.process(audio_chunk, sample_rate=48000)
print(gc.get_state())
# {'mode': 'agc', 'current_gain_db': 3.2, 'noise_gate_enabled': True, ...}
```

---

### 5. RVCEngine (`ai-engine/rvc/rvc_engine.py`)

Core voice conversion engine. Loads model checkpoints, extracts features and pitch, runs neural synthesis, and applies pitch shifting.

| Method | Description |
|---|---|
| `load_model(path)` | Load a .pth model checkpoint (with optional .index file) |
| `set_pitch_shift(semitones)` | Set pitch shift (-24 to +24) |
| `convert_audio(audio, voice, pitch, sr)` | Convert audio chunk to target voice |
| `get_available_voices()` | List all .pth files in models/presets/ |
| `get_device_info()` | Get GPU/CPU device status and memory usage |

**Conversion pipeline:**
1. Stereo→mono downmix
2. Content feature extraction (FeatureExtractor)
3. F0 pitch extraction (PitchExtractor)
4. Pitch shifting (semitone offset)
5. Neural synthesis (SyntheticRVCSynthesizer)
6. Output resampling to match input length
7. Normalization to prevent clipping
8. Channel layout restoration

```python
from ai_engine.rvc.rvc_engine import RVCEngine

engine = RVCEngine(default_pitch_shift=-5.0)
engine.load_model("ai-engine/models/presets/anime_heroine.pth")
converted = engine.convert_audio(audio_chunk, sample_rate=48000)
```

**Device selection:** Automatically uses CUDA GPU if available, falls back to CPU.

---

### 6. RealtimeRVCInference (`ai-engine/rvc/rvc_inference.py`)

The streaming worker thread that manages audio I/O, circular buffers, and the real-time processing loop.

**Processing loop stages (per chunk):**
1. Read chunk from input circular buffer
2. Stereo→mono downmix
3. Noise suppression (noisereduce)
4. Gain control (AGC/manual)
5. RVC voice conversion
6. Audio effects (pedalboard: compressor, EQ, reverb, limiter)
7. Mono→stereo restoration
8. Write to output circular buffer

**Latency tracking:** Exponential moving average (EMA) with α=0.1 for smoothing.

```python
from ai_engine.rvc.rvc_inference import RealtimeRVCInference

worker = RealtimeRVCInference(
    rvc_engine=engine,
    noise_suppressor=ns,
    effects_chain=fx,
    gain_controller=gc,
    sample_rate=48000,
    chunk_size=2048,
    buffer_size=8192
)
worker.start()
# ... worker.running = True ...
worker.stop()
```

**Push-to-talk:** When PTT is enabled, audio is only captured when `ptt_active` is True. This allows keyboard hotkey binding from the frontend.

---

### 7. CircularAudioBuffer (`ai-engine/inference/audio_buffer.py`)

Thread-safe ring buffer for low-latency audio streaming between the audio callback thread and processing thread.

| Method | Description |
|---|---|
| `write(data)` | Write audio frames (overwrites oldest if full) |
| `read(n, fill_padding)` | Read up to n frames (optionally zero-pad) |
| `clear()` | Reset buffer state |
| `available_read()` | Frames available for reading |
| `available_write()` | Remaining capacity |
| `is_full()` / `is_empty()` | Buffer state checks |

**Key design:**
- `threading.Condition` for blocking synchronization
- Supports 1D (mono) and 2D (multi-channel) arrays
- Automatic overflow handling (drops oldest unread frames)
- NumPy float32 throughout

---

### 8. AudioEffectsChain (`backend/audio_effects.py`)

DSP post-processing using Spotify's Pedalboard library.

| Effect | Parameters |
|---|---|
| Compressor | threshold_db, ratio, attack_ms, release_ms |
| De-esser | threshold_db, frequency_hz (implemented as HighShelfFilter) |
| Parametric EQ | low_gain_db, mid_gain_db, high_gain_db (3-band) |
| Reverb | room_size, wet_level, dry_level, damping |
| Limiter | threshold_db (brickwall) |

```python
from backend.audio_effects import AudioEffectsChain

fx = AudioEffectsChain()
fx.update_config({
    "reverb": {"enabled": True, "room_size": 0.35, "wet_level": 0.20},
    "eq": {"mid_gain_db": 2.5, "high_gain_db": 3.0}
})
processed = fx.process(audio_chunk, sample_rate=48000)
```

---

### 9. NoiseSuppressor (`backend/noise_suppression.py`)

Background noise removal using noisereduce spectral gating.

| Parameter | Default | Description |
|---|---|---|
| `enabled` | True | Enable/disable noise suppression |
| `mode` | "spectral_gating" | Algorithm mode |
| `prop_decrease` | 0.85 | Proportion of noise to reduce (0-1) |
| `stationary` | True | Assume stationary noise |
| `n_fft` | 1024 | FFT window size |
| `time_mask_smooth_ms` | 50 | Time mask smoothing |

Falls back to a simple noise gate threshold when noisereduce is not installed.

---

### 10. AudioRouter (`app/routing/audio_router.py`)

Routes processed audio to virtual audio devices for OBS, Discord, Zoom, VRChat, etc.

**Target profiles:**

| Target | Sample Rate | Buffer | Channels | Use Case |
|---|---|---|---|---|
| `obs` | 48000 | 512 | 2 (stereo) | High-fidelity streaming/recording |
| `discord` | 48000 | 1024 | 1 (mono) | VoIP communication |
| `zoom` | 44100 | 1024 | 1 (mono) | Conference calls |
| `vrchat` | 48000 | 256 | 1 (mono) | Ultra-low latency gaming |

**Platform-specific virtual devices:**
- **Windows:** VB-Audio Virtual Cable
- **macOS:** BlackHole 2ch
- **Linux:** PipeWire/PulseAudio null sink

```python
from app.routing.audio_router import AudioRouter

router = AudioRouter()
router.initialize_router("discord")
router.route_audio(processed_chunk)
status = router.get_routing_status()
```

---

## Configuration

### `configs/default.yaml`

```yaml
audio:
  sample_rate: 48000
  chunk_size: 2048
  buffer_size: 4096
  channels: 1
  input_device: null    # null = system default
  output_device: null

engine:
  model_path: "ai-engine/models/presets/default.pth"
  device: "auto"        # "auto", "cuda", or "cpu"
  pitch_shift: 0.0
  index_rate: 0.75
  filter_radius: 3
  resample_sr: 0

noise_suppression:
  enabled: true
  mode: "spectral_gating"
  prop_decrease: 0.85
  stationary: true

effects:
  enabled: true
  compressor:
    enabled: true
    threshold_db: -16.0
    ratio: 3.0
    attack_ms: 10.0
    release_ms: 100.0
  de_esser:
    enabled: true
    threshold_db: -20.0
    frequency_hz: 6000.0
  eq:
    enabled: true
    low_gain_db: 0.0
    mid_gain_db: 1.5
    high_gain_db: 2.0
  reverb:
    enabled: false
    room_size: 0.25
    wet_level: 0.15
    dry_level: 0.85
    damping: 0.5
  limiter:
    enabled: true
    threshold_db: -1.0
    release_ms: 50.0

server:
  host: "0.0.0.0"
  port: 7860
  cors_origins: ["*"]
```

### Runtime Settings Update

Settings can be updated at runtime via `update_settings()`:

```python
pipeline.update_settings({
    "pitch_shift": -5.0,
    "noise_suppression": {"prop_decrease": 0.9},
    "effects": {"reverb": {"enabled": True, "room_size": 0.4}},
    "gain": {"mode": "agc", "manual_gain_db": 3.0},
    "dry_wet": 0.85,
    "bypass": False,
    "ptt_enabled": True,
    "input_device": 2,
    "output_device": 5,
    "recording": {"enabled": True, "output_path": "recordings/session1.wav"}
})
```

---

## REST API

All endpoints are served by the FastAPI backend at `http://localhost:7860`.
Interactive Swagger docs available at `http://localhost:7860/docs`.

### Health & System

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Server health, pipeline status, device info |
| GET | `/api/voices` | List available voice presets |

### Conversion Control

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/conversion/toggle` | Start/stop voice conversion |
| POST | `/api/conversion/preset` | Switch voice preset (+ optional pitch) |
| GET | `/api/conversion/status` | Full status: state, latency, levels, device, routing |

### Audio Configuration

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/audio/devices` | List system audio input/output devices |
| POST | `/api/audio/settings` | Update pitch, noise, effects, gain, devices, PTT |
| GET | `/api/audio/levels` | Get real-time input/output RMS + dB levels |

### Gain Control

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/gain` | Get gain controller state |
| POST | `/api/gain` | Update gain mode, manual gain, noise gate |

### Push-to-Talk

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/ptt/toggle` | Enable/disable PTT mode |
| POST | `/api/ptt/active` | Set PTT transmit state (active=True/False) |

### Example: Start conversion and set pitch

```bash
# Start the pipeline
curl -X POST http://localhost:7860/api/conversion/toggle \
  -H "Content-Type: application/json" \
  -d '{"active": true}'

# Switch to a voice preset with pitch shift
curl -X POST http://localhost:7860/api/conversion/preset \
  -H "Content-Type: application/json" \
  -d '{"preset_id": "anime_heroine", "pitch_shift": -5.0}'

# Update audio settings
curl -X POST http://localhost:7860/api/audio/settings \
  -H "Content-Type: application/json" \
  -d '{
    "pitch_shift": -5.0,
    "noise_suppression": {"prop_decrease": 0.9},
    "gain": {"mode": "agc"},
    "dry_wet": 0.85
  }'
```

---

## WebSocket API

### `ws://localhost:7860/ws/levels`

Streams real-time audio levels and telemetry at ~20 FPS.

**Message format (server → client):**
```json
{
  "input_rms": 0.0342,
  "output_rms": 0.0518,
  "input_db": -29.32,
  "output_db": -25.71,
  "is_running": true,
  "ptt_enabled": false,
  "ptt_active": false,
  "latency": {
    "processing_ms": 12.4,
    "buffer_ms": 42.67,
    "total_ms": 55.07,
    "frames_processed": 1842,
    "frames_dropped": 0
  },
  "active_preset": "anime_heroine"
}
```

---

## Latency Budget

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    LATENCY BUDGET (48kHz, 2048 chunk)                    │
├───────────────────────┬──────────────┬───────────────────────────────────┤
│ Stage                 │ Duration     │ Notes                             │
├───────────────────────┼──────────────┼───────────────────────────────────┤
│ Buffer (chunk_size)   │ 42.67 ms     │ 2048 / 48000 × 1000              │
│ Noise Suppression     │ 1-3 ms       │ noisereduce spectral gating       │
│ Gain Control          │ <0.5 ms      │ NumPy multiply                    │
│ Feature Extraction    │ 2-8 ms       │ Conv1d encoder (GPU) / FFT (CPU)  │
│ Pitch Extraction      │ 1-5 ms       │ Autocorrelation (fast) / pyin     │
│ RVC Neural Synthesis  │ 5-20 ms      │ GPU dependent, CPU fallback slow  │
│ Audio Effects         │ 1-3 ms       │ Pedalboard (native C++)           │
│ Virtual Mic Output    │ 5-10 ms      │ Buffer + device write             │
├───────────────────────┼──────────────┼───────────────────────────────────┤
│ TOTAL (GPU)           │ ~58-92 ms    │ Acceptable for real-time use       │
│ TOTAL (CPU fallback)  │ ~120-200 ms  │ Noticeable delay                  │
└───────────────────────┴──────────────┴───────────────────────────────────┘
```

**Optimization tips:**
- Reduce `chunk_size` to 1024 or 512 for lower buffer latency (increases CPU overhead)
- Use CUDA GPU for feature extraction and neural synthesis
- Disable reverb effect if not needed (saves ~1-2ms)
- Use `autocorrelation` pitch method instead of `pyin` for faster F0 estimation
- Increase `buffer_size` to reduce underruns at the cost of higher latency

---

## Dependency Matrix

| Package | Required | Purpose |
|---|---|---|
| `torch` | Yes | Neural network inference (RVC engine + feature extraction) |
| `numpy` | Yes | Audio array processing |
| `scipy` | Yes | Signal processing (resampling, interpolation) |
| `sounddevice` | Yes | Audio I/O (PortAudio bindings) |
| `fastapi` | Yes | REST API server |
| `uvicorn` | Yes | ASGI server |
| `pyyaml` | Recommended | YAML config loading |
| `noisereduce` | Recommended | Noise suppression (falls back to noise gate) |
| `pedalboard` | Recommended | Audio effects (falls back to software limiter) |
| `librosa` | Optional | High-accuracy pyin pitch extraction |
| `fairseq` | Optional | Full HuBERT model via torch.hub |

Install all dependencies:
```bash
pip install torch numpy scipy sounddevice fastapi uvicorn pyyaml noisereduce pedalboard librosa
```

---

## Troubleshooting

### No audio output / "Could not open audio streams"
- Check that your input/output devices are not in use by another application
- Verify virtual audio cable is installed (VB-Cable/BlackHole/PipeWire)
- List available devices: `python -c "import sounddevice as sd; print(sd.query_devices())"`

### High latency (>100ms)
- Check if GPU is being used: `GET /api/health` → `device_info.device` should show `cuda`
- Reduce `chunk_size` in config to 1024 or 512
- Disable reverb and reduce effects complexity
- Use `autocorrelation` pitch method

### Audio glitches / dropouts
- Check buffer health: `GET /api/conversion/status` → `buffer_health`
- Increase `buffer_size` in config
- Close CPU-intensive applications
- Ensure GPU has sufficient free VRAM

### Model loading fails
- Verify `.pth` file exists in `ai-engine/models/presets/`
- Check file is a valid PyTorch checkpoint
- Look for companion `.index` file (optional but recommended)
- Pipeline falls back to default synth model if loading fails (non-fatal)
