# FastAPI Backend API Reference

The **Real-Time AI Voice Changer** control plane is driven by a high-performance FastAPI service running at default host `http://127.0.0.1:8000`.

Interactive Swagger UI documentation is available at `http://127.0.0.1:8000/docs` when the server is running.

---

## Base URL
```text
http://127.0.0.1:8000/api/v1
```

---

## 1. System & Health Endpoints

### `GET /system/status`
Returns general application health, current processing state, active voice model, and hardware resource utilization.

#### Request
```http
GET /api/v1/system/status HTTP/1.1
Host: 127.0.0.1:8000
Accept: application/json
```

#### Response (`200 OK`)
```json
{
  "status": "active",
  "version": "1.2.0",
  "uptime_seconds": 3412.5,
  "engine_running": true,
  "current_model": "anime_heroine_v2",
  "active_device": "cuda:0",
  "gpu_info": {
    "name": "NVIDIA GeForce RTX 4070",
    "vram_used_mb": 2145,
    "vram_total_mb": 12282,
    "gpu_utilization_pct": 28.5
  },
  "audio": {
    "sample_rate": 48000,
    "buffer_size": 512,
    "input_device": "Elgato Wave:3 (Hardware)",
    "output_device": "CABLE Input (VB-Audio Virtual Cable)"
  }
}
```

---

## 2. Voice Model Management Endpoints

### `GET /models`
Lists all available trained voice conversion models (`.pth` / `.onnx` / `.engine`) and index files found in the `models/weights` folder.

#### Request
```http
GET /api/v1/models HTTP/1.1
```

#### Response (`200 OK`)
```json
{
  "models": [
    {
      "id": "anime_heroine_v2",
      "name": "Anime Heroine V2",
      "format": "pth",
      "sample_rate": 48000,
      "has_index": true,
      "index_file": "anime_heroine_v2.index",
      "size_bytes": 55482104,
      "created_at": "2026-08-01T14:22:10Z"
    },
    {
      "id": "deep_announcer_trt",
      "name": "Deep Announcer (TensorRT FP16)",
      "format": "engine",
      "sample_rate": 40000,
      "has_index": false,
      "index_file": null,
      "size_bytes": 38910240,
      "created_at": "2026-08-05T09:15:00Z"
    }
  ]
}
```

---

### `POST /models/select`
Hot-swaps the currently active voice conversion model in real time without stopping the stream.

#### Request
```http
POST /api/v1/models/select HTTP/1.1
Content-Type: application/json

{
  "model_id": "deep_announcer_trt",
  "index_rate": 0.65,
  "pitch_algorithm": "rmvpe"
}
```

#### Response (`200 OK`)
```json
{
  "success": true,
  "message": "Voice model switched to 'deep_announcer_trt' in 142ms",
  "active_model": {
    "id": "deep_announcer_trt",
    "format": "engine",
    "sample_rate": 40000,
    "pitch_algorithm": "rmvpe"
  }
}
```

---

### `POST /models/upload`
Uploads new `.pth` or `.index` weight files into the server repository.

#### Request
`multipart/form-data` with fields `model_file` and optional `index_file`.

#### Response (`201 Created`)
```json
{
  "success": true,
  "model_id": "custom_voice_2026",
  "files_saved": [
    "models/weights/custom_voice_2026.pth",
    "models/weights/custom_voice_2026.index"
  ]
}
```

---

## 3. Audio & DSP Configuration Endpoints

### `GET /audio/config`
Retrieves current audio driver mappings, pitch shift, gain levels, and active DSP filters.

#### Request
```http
GET /api/v1/audio/config HTTP/1.1
```

#### Response (`200 OK`)
```json
{
  "input_device_id": 1,
  "output_device_id": 3,
  "sample_rate": 48000,
  "buffer_size": 512,
  "pitch_shift_semitones": 3,
  "formant_shift": 1.05,
  "input_gain_db": 0.0,
  "output_gain_db": 2.5,
  "dsp_filters": {
    "noise_suppression": true,
    "noise_threshold_db": -45.0,
    "eq_enabled": true,
    "eq_bands_db": [0.0, 1.5, -2.0, 3.0, 0.5],
    "compressor_enabled": true,
    "reverb_mix": 0.08
  }
}
```

---

### `PUT /audio/config`
Updates active DSP, pitch shift, or gain configuration in real time.

#### Request
```http
PUT /api/v1/audio/config HTTP/1.1
Content-Type: application/json

{
  "pitch_shift_semitones": 5,
  "formant_shift": 1.10,
  "output_gain_db": 1.5,
  "dsp_filters": {
    "noise_suppression": true,
    "reverb_mix": 0.12
  }
}
```

#### Response (`200 OK`)
```json
{
  "success": true,
  "updated_config": {
    "pitch_shift_semitones": 5,
    "formant_shift": 1.10,
    "output_gain_db": 1.5,
    "dsp_filters": {
      "noise_suppression": true,
      "reverb_mix": 0.12
    }
  }
}
```

---

## 4. Stream Control & Latency Stats Endpoints

### `POST /stream/start`
Starts real-time audio capture, background worker threads, and virtual driver output routing.

#### Request
```http
POST /api/v1/stream/start HTTP/1.1
```

#### Response (`200 OK`)
```json
{
  "status": "running",
  "started_at": "2026-08-09T02:45:00Z",
  "input_device": "Elgato Wave:3",
  "output_device": "CABLE Input"
}
```

---

### `POST /stream/stop`
Stops stream processing and releases hardware/virtual audio interfaces.

#### Request
```http
POST /api/v1/stream/stop HTTP/1.1
```

#### Response (`200 OK`)
```json
{
  "status": "stopped",
  "stopped_at": "2026-08-09T02:46:12Z"
}
```

---

### `GET /stream/telemetry`
Returns real-time audio latency benchmarks, frame drop counts, and signal levels.

#### Request
```http
GET /api/v1/stream/telemetry HTTP/1.1
```

#### Response (`200 OK`)
```json
{
  "timestamp": 1786243512.41,
  "input_rms_db": -22.4,
  "output_rms_db": -18.1,
  "latency_ms": {
    "audio_capture": 5.33,
    "noise_suppression": 1.82,
    "pitch_extraction": 6.10,
    "model_inference": 14.25,
    "post_dsp": 1.10,
    "buffer_out": 5.33,
    "total_end_to_end": 33.93
  },
  "buffer_overruns": 0,
  "buffer_underruns": 0,
  "dropped_frames": 0
}
```

---

## 5. WebSocket API

### `WS /ws/stream`
Full-duplex WebSocket connection for streaming client control and receiving sub-second audio telemetry.

#### Connection String
```text
ws://127.0.0.1:8000/api/v1/ws/stream
```

#### Telemetry Event (Server -> Client @ 30Hz)
```json
{
  "event": "telemetry",
  "data": {
    "input_peak": 0.42,
    "output_peak": 0.68,
    "latency_ms": 34.2,
    "gpu_temp_c": 62,
    "vad_active": true
  }
}
```

#### Parameter Control Message (Client -> Server)
```json
{
  "action": "set_pitch",
  "semitones": -2
}
```

#### Response Acknowledgement (Server -> Client)
```json
{
  "event": "ack",
  "action": "set_pitch",
  "value": -2,
  "timestamp": 1786243515.10
}
```
