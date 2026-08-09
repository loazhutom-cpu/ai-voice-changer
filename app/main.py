"""
FastAPI Backend Application Entrypoint for AI Voice Changer.
"""

from fastapi import FastAPI
from app.routing.audio_router import AudioRouter
from app.presets.preset_manager import PresetManager

app = FastAPI(title="AI Voice Changer Backend API", version="1.0.0")

router_instance = AudioRouter()
preset_manager = PresetManager()

@app.get("/")
def read_root():
    return {"status": "ok", "message": "AI Voice Changer Backend Operational"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/api/presets")
def get_presets():
    return preset_manager.list_presets()

@app.get("/api/routing/status")
def get_routing_status():
    return router_instance.get_routing_status()
