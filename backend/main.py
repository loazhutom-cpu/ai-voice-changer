"""
FastAPI Server for AI Voice Conversion Engine.

Exposes REST endpoints and WebSockets for managing voice conversion, changing presets,
configuring audio hardware, adjusting effects, gain control, push-to-talk,
and streaming real-time audio meters.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import sounddevice as sd
    HAS_SD = True
except ImportError:
    HAS_SD = False

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_engine.inference.pipeline import AudioPipeline
from ai_engine.models.voice_models import VoiceModelRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("voice_changer_backend")

# Initialize FastAPI
app = FastAPI(
    title="AI Voice Changer Engine",
    description="Real-time RVC voice conversion backend server",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
pipeline = AudioPipeline()
voice_registry = VoiceModelRegistry()


# ─── Pydantic Schemas ──────────────────────────────────────────────

class ToggleConversionRequest(BaseModel):
    active: Optional[bool] = Field(default=None, description="True=start, False=stop, omit=toggle")


class SwitchPresetRequest(BaseModel):
    preset_id: str = Field(..., description="Target voice preset ID")
    pitch_shift: Optional[float] = Field(default=None, description="Pitch shift in semitones")


class AudioSettingsRequest(BaseModel):
    pitch_shift: Optional[float] = Field(default=None, description="Semitones offset (-24 to +24)")
    input_device: Optional[Union[int, str]] = Field(default=None)
    output_device: Optional[Union[int, str]] = Field(default=None)
    noise_suppression: Optional[Any] = Field(default=None)
    effects: Optional[Dict[str, Any]] = Field(default=None)
    gain: Optional[Any] = Field(default=None, description="Gain config dict or dB value")
    ptt_enabled: Optional[bool] = Field(default=None, description="Enable push-to-talk")


class PTTRequest(BaseModel):
    active: bool = Field(..., description="True=transmitting, False=muted")


class GainSettingsRequest(BaseModel):
    mode: Optional[str] = Field(default=None, description="'manual', 'agc', or 'bypass'")
    manual_gain_db: Optional[float] = Field(default=None)
    noise_gate_enabled: Optional[bool] = Field(default=None)


# ─── REST Endpoints ─────────────────────────────────────────────────

@app.get("/api/health", summary="Health Check")
def get_health() -> Dict[str, Any]:
    """Check server status and active components."""
    return {
        "status": "healthy",
        "service": "ai-voice-changer-engine",
        "version": "1.1.0",
        "pipeline_running": pipeline.is_running(),
        "active_preset": pipeline.current_preset,
        "device_info": pipeline.get_device_info()
    }


@app.get("/api/voices", summary="List Voice Presets")
def get_voices() -> Dict[str, Any]:
    """List all available voice model presets."""
    voices = voice_registry.list_voices()
    return {
        "count": len(voices),
        "voices": voices,
        "active_preset": pipeline.current_preset
    }


@app.post("/api/conversion/toggle", summary="Start/Stop Conversion")
def toggle_conversion(request: Optional[ToggleConversionRequest] = None) -> Dict[str, Any]:
    """Start or stop real-time voice conversion."""
    is_running = pipeline.is_running()
    target = not is_running if (request is None or request.active is None) else request.active

    if target:
        res = pipeline.start()
    else:
        res = pipeline.stop()

    return {"is_running": pipeline.is_running(), "details": res}


@app.post("/api/conversion/preset", summary="Switch Voice Preset")
def switch_preset(request: SwitchPresetRequest) -> Dict[str, Any]:
    """Switch voice preset and optionally set pitch shift."""
    success = pipeline.set_voice_preset(request.preset_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Voice preset '{request.preset_id}' not found.")

    if request.pitch_shift is not None:
        pipeline.set_pitch_shift(request.pitch_shift)

    return {
        "status": "success",
        "active_preset": pipeline.current_preset,
        "pitch_shift": pipeline.rvc_engine.pitch_shift
    }


@app.get("/api/conversion/status", summary="Get Conversion Status")
def get_conversion_status() -> Dict[str, Any]:
    """Get status, latency, and audio levels."""
    return {
        "is_running": pipeline.is_running(),
        "active_preset": pipeline.current_preset,
        "pitch_shift": pipeline.rvc_engine.pitch_shift,
        "latency": pipeline.get_latency(),
        "levels": pipeline.get_audio_levels(),
        "device_info": pipeline.get_device_info()
    }


@app.get("/api/audio/devices", summary="List Audio Devices")
def list_audio_devices() -> Dict[str, Any]:
    """List available input/output audio devices."""
    if not HAS_SD:
        return {"count": 0, "devices": [], "error": "sounddevice not available"}

    try:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        device_list = []

        for idx, dev in enumerate(devices):
            device_list.append({
                "id": idx,
                "name": dev["name"],
                "max_input_channels": dev["max_input_channels"],
                "max_output_channels": dev["max_output_channels"],
                "default_samplerate": dev["default_samplerate"],
                "hostapi": hostapis[dev["hostapi"]]["name"] if dev["hostapi"] < len(hostapis) else "Unknown"
            })

        return {
            "count": len(device_list),
            "devices": device_list,
            "default_input": sd.default.device[0],
            "default_output": sd.default.device[1]
        }
    except Exception as e:
        logger.error(f"Error querying audio devices: {e}")
        return {"count": 0, "devices": [], "error": str(e)}


@app.post("/api/audio/settings", summary="Update Audio Settings")
def update_audio_settings(request: AudioSettingsRequest) -> Dict[str, Any]:
    """Update pitch, noise suppression, effects, gain, devices, PTT."""
    settings_dict = request.model_dump(exclude_unset=True)
    updated = pipeline.update_settings(settings_dict)
    return {"status": "success", "updated_settings": updated}


@app.get("/api/audio/levels", summary="Get Audio Levels")
def get_audio_levels() -> Dict[str, Any]:
    """Get real-time input/output RMS and dB levels."""
    return pipeline.get_audio_levels()


# ─── Push-to-Talk ───────────────────────────────────────────────────

@app.post("/api/ptt/toggle", summary="Enable/Disable Push-to-Talk")
def toggle_ptt_enabled() -> Dict[str, Any]:
    """Toggle push-to-talk mode on/off."""
    current = pipeline.inference_worker._ptt_enabled
    pipeline.set_ptt_enabled(not current)
    return {"ptt_enabled": pipeline.inference_worker._ptt_enabled}


@app.post("/api/ptt/active", summary="Set PTT Transmit State")
def set_ptt_active(request: PTTRequest) -> Dict[str, Any]:
    """Set push-to-talk button state (True=transmit, False=mute)."""
    pipeline.set_ptt_active(request.active)
    return {
        "ptt_active": request.active,
        "ptt_enabled": pipeline.inference_worker._ptt_enabled
    }


# ─── Gain Control ──────────────────────────────────────────────────

@app.get("/api/gain", summary="Get Gain State")
def get_gain_state() -> Dict[str, Any]:
    """Get current gain controller state."""
    return pipeline.gain_controller.get_state()


@app.post("/api/gain", summary="Update Gain Settings")
def update_gain_settings(request: GainSettingsRequest) -> Dict[str, Any]:
    """Update gain mode, manual gain, and noise gate settings."""
    if request.mode is not None:
        pipeline.gain_controller.set_mode(request.mode)
    if request.manual_gain_db is not None:
        pipeline.gain_controller.set_manual_gain(request.manual_gain_db)
    if request.noise_gate_enabled is not None:
        pipeline.gain_controller.noise_gate_enabled = request.noise_gate_enabled

    return {"status": "success", "gain_state": pipeline.gain_controller.get_state()}


# ─── WebSocket ──────────────────────────────────────────────────────

@app.websocket("/ws/levels")
async def websocket_audio_levels(websocket: WebSocket):
    """Stream real-time audio levels via WebSocket at ~20 FPS."""
    await websocket.accept()
    logger.info("WebSocket client connected for audio levels.")
    try:
        while True:
            levels = pipeline.get_audio_levels()
            levels["latency"] = pipeline.get_latency()
            levels["active_preset"] = pipeline.current_preset
            await websocket.send_json(levels)
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=7860, reload=True)
