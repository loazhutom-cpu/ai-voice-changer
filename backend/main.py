"""
FastAPI Server for AI Voice Conversion Engine.

Exposes REST endpoints and WebSockets for managing voice conversion, changing presets,
configuring audio hardware, adjusting pedalboard effects, and streaming real-time audio meters.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import sounddevice as sd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_engine.inference.pipeline import AudioPipeline
from ai_engine.models.voice_models import VoiceModelRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("voice_changer_backend")

# Initialize FastAPI App
app = FastAPI(
    title="AI Voice Changer Engine",
    description="Real-time RVC voice conversion backend server",
    version="1.0.0"
)

# CORS Middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Pipeline and Registry instances
pipeline = AudioPipeline()
voice_registry = VoiceModelRegistry()


# Pydantic Request & Response Schemas
class ToggleConversionRequest(BaseModel):
    active: Optional[bool] = Field(default=None, description="Set True to start conversion, False to stop. If omitted, toggles status.")


class SwitchPresetRequest(BaseModel):
    preset_id: str = Field(..., description="Target voice preset identifier")
    pitch_shift: Optional[float] = Field(default=None, description="Optional pitch shift adjustment in semitones")


class AudioSettingsRequest(BaseModel):
    pitch_shift: Optional[float] = Field(default=None, description="Semitones offset (-24 to +24)")
    input_device: Optional[Union[int, str]] = Field(default=None, description="Input device index or string name")
    output_device: Optional[Union[int, str]] = Field(default=None, description="Output device index or string name")
    noise_suppression: Optional[Dict[str, Any]] = Field(default=None, description="Noise suppression configuration dict or boolean")
    effects: Optional[Dict[str, Any]] = Field(default=None, description="Audio effects configuration dict")


# REST Endpoints
@app.get("/api/health", summary="Health Check")
def get_health() -> Dict[str, Any]:
    """Check API server operational status and active components."""
    return {
        "status": "healthy",
        "service": "ai-voice-changer-engine",
        "version": "1.0.0",
        "pipeline_running": pipeline.is_running(),
        "active_preset": pipeline.current_preset
    }


@app.get("/api/voices", summary="List Available Voice Presets")
def get_voices() -> Dict[str, Any]:
    """List all registered and scanned voice model presets."""
    voices = voice_registry.list_voices()
    return {
        "count": len(voices),
        "voices": voices,
        "active_preset": pipeline.current_preset
    }


@app.post("/api/conversion/toggle", summary="Start/Stop Voice Conversion")
def toggle_conversion(request: Optional[ToggleConversionRequest] = None) -> Dict[str, Any]:
    """Start or stop real-time voice conversion audio pipeline."""
    is_currently_running = pipeline.is_running()
    target_active = not is_currently_running if (request is None or request.active is None) else request.active

    if target_active:
        res = pipeline.start()
    else:
        res = pipeline.stop()

    return {
        "is_running": pipeline.is_running(),
        "details": res
    }


@app.post("/api/conversion/preset", summary="Switch Voice Preset")
def switch_preset(request: SwitchPresetRequest) -> Dict[str, Any]:
    """Switch target voice model preset and optionally set pitch shift."""
    success = pipeline.set_voice_preset(request.preset_id)
    if not success:
        raise HTTPException(status_code=44, detail=f"Voice preset '{request.preset_id}' not found or invalid.")

    if request.pitch_shift is not None:
        pipeline.set_pitch_shift(request.pitch_shift)

    return {
        "status": "success",
        "active_preset": pipeline.current_preset,
        "pitch_shift": pipeline.rvc_engine.pitch_shift
    }


@app.get("/api/conversion/status", summary="Get Conversion Status")
def get_conversion_status() -> Dict[str, Any]:
    """Get conversion status, latency breakdown, and current audio levels."""
    return {
        "is_running": pipeline.is_running(),
        "active_preset": pipeline.current_preset,
        "pitch_shift": pipeline.rvc_engine.pitch_shift,
        "latency": pipeline.get_latency(),
        "levels": pipeline.get_audio_levels()
    }


@app.get("/api/audio/devices", summary="List Audio Devices")
def list_audio_devices() -> Dict[str, Any]:
    """List available input and output host audio devices."""
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
        logger.error(f"Error querying sounddevice audio devices: {e}")
        return {
            "count": 0,
            "devices": [],
            "error": str(e)
        }


@app.post("/api/audio/settings", summary="Update Audio Settings")
def update_audio_settings(request: AudioSettingsRequest) -> Dict[str, Any]:
    """Update settings including pitch shift, noise suppression, and pedalboard effects."""
    settings_dict = request.model_dump(exclude_unset=True)
    updated_info = pipeline.update_settings(settings_dict)
    return {
        "status": "success",
        "updated_settings": updated_info
    }


@app.get("/api/audio/levels", summary="Get Audio Levels")
def get_audio_levels() -> Dict[str, Any]:
    """Get real-time input and output RMS levels and dB values."""
    return pipeline.get_audio_levels()


# WebSocket Endpoint
@app.websocket("/ws/levels")
async def websocket_audio_levels(websocket: WebSocket):
    """
    WebSocket streaming real-time audio levels (RMS/dB) every 50ms.
    """
    await websocket.accept()
    logger.info("WebSocket connection established for audio level streaming.")
    try:
        while True:
            levels = pipeline.get_audio_levels()
            levels["is_running"] = pipeline.is_running()
            await websocket.send_json(levels)
            await asyncio.sleep(0.05)  # 20 FPS meter updates
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=7860, reload=True)
