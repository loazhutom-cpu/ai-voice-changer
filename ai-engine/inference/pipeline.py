"""
Main Audio Processing Pipeline Orchestrator.

Chains noise suppression, gain control, RVC voice conversion, pedalboard effects,
and low-latency audio routing. Provides a unified API for start/stop, preset switching,
settings updates, and real-time status queries.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from ai_engine.inference.gain_control import GainController
    from ai_engine.rvc.rvc_engine import RVCEngine
    from ai_engine.rvc.rvc_inference import RealtimeRVCInference
except ImportError:
    try:
        from ..inference.gain_control import GainController
        from ..rvc.rvc_engine import RVCEngine
        from ..rvc.rvc_inference import RealtimeRVCInference
    except ImportError:
        from inference.gain_control import GainController
        from rvc.rvc_engine import RVCEngine
        from rvc.rvc_inference import RealtimeRVCInference

from backend.audio_effects import AudioEffectsChain
from backend.noise_suppression import NoiseSuppressor

logger = logging.getLogger(__name__)


class AudioPipeline:
    """
    Main audio processing pipeline orchestrator.

    Integrates RVC Engine, Noise Suppressor, Gain Controller, Audio Effects,
    and real-time streaming into a unified control interface.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        sample_rate: int = 48000,
        chunk_size: int = 2048,
        buffer_size: int = 8192
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size

        # Core components
        self.rvc_engine = RVCEngine()
        self.noise_suppressor = NoiseSuppressor()
        self.effects_chain = AudioEffectsChain()
        self.gain_controller = GainController()

        # Real-time streaming worker
        self.inference_worker = RealtimeRVCInference(
            rvc_engine=self.rvc_engine,
            noise_suppressor=self.noise_suppressor,
            effects_chain=self.effects_chain,
            gain_controller=self.gain_controller,
            sample_rate=self.sample_rate,
            chunk_size=self.chunk_size,
            buffer_size=self.buffer_size
        )

        self.current_preset: str = "default_voice"
        logger.info("AudioPipeline orchestrator initialized.")

    def start(self) -> Dict[str, Any]:
        """Start the real-time pipeline."""
        logger.info("Starting AudioPipeline...")
        self.inference_worker.start()
        return {
            "status": "running",
            "active_preset": self.current_preset,
            "sample_rate": self.sample_rate,
            "chunk_size": self.chunk_size
        }

    def stop(self) -> Dict[str, Any]:
        """Stop the real-time pipeline."""
        logger.info("Stopping AudioPipeline...")
        self.inference_worker.stop()
        return {"status": "stopped"}

    def is_running(self) -> bool:
        """Check if pipeline is active."""
        return self.inference_worker.is_running()

    def set_voice_preset(self, preset_id: str) -> bool:
        """Switch active voice model preset."""
        logger.info(f"Switching voice preset to: {preset_id}")
        if preset_id == "default_voice":
            self.current_preset = preset_id
            return True
        success = self.rvc_engine.load_model(f"ai-engine/models/presets/{preset_id}.pth")
        if success:
            self.current_preset = preset_id
            return True
        return False

    def set_pitch_shift(self, semitones: float) -> None:
        """Set pitch shift in semitones (-24 to +24)."""
        self.rvc_engine.set_pitch_shift(semitones)

    def get_latency(self) -> Dict[str, float]:
        """Get latency breakdown in milliseconds."""
        return self.inference_worker.get_latency_stats()

    def get_audio_levels(self) -> Dict[str, float]:
        """Get current audio levels for UI meters."""
        return self.inference_worker.get_audio_levels()

    def get_available_voices(self) -> List[Dict[str, Any]]:
        """List available voice presets."""
        return self.rvc_engine.get_available_voices()

    def get_device_info(self) -> Dict[str, Any]:
        """Get GPU/device diagnostics."""
        return self.rvc_engine.get_device_info()

    def set_ptt_enabled(self, enabled: bool) -> None:
        """Enable or disable push-to-talk mode."""
        self.inference_worker.set_ptt_enabled(enabled)

    def set_ptt_active(self, active: bool) -> None:
        """Set push-to-talk transmit state."""
        self.inference_worker.set_ptt_active(active)

    def update_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update runtime audio settings.

        Supports: pitch_shift, noise_suppression, effects, gain, input_device,
        output_device, ptt_enabled.
        """
        if "pitch_shift" in settings:
            self.set_pitch_shift(float(settings["pitch_shift"]))

        if "noise_suppression" in settings:
            ns_cfg = settings["noise_suppression"]
            if isinstance(ns_cfg, dict):
                self.noise_suppressor.update_config(ns_cfg)
            elif isinstance(ns_cfg, bool):
                self.noise_suppressor.enabled = ns_cfg

        if "effects" in settings:
            self.effects_chain.update_config(settings["effects"])

        if "gain" in settings:
            gain_cfg = settings["gain"]
            if isinstance(gain_cfg, dict):
                if "mode" in gain_cfg:
                    self.gain_controller.set_mode(gain_cfg["mode"])
                if "manual_gain_db" in gain_cfg:
                    self.gain_controller.set_manual_gain(gain_cfg["manual_gain_db"])
                if "noise_gate_enabled" in gain_cfg:
                    self.gain_controller.noise_gate_enabled = bool(gain_cfg["noise_gate_enabled"])
            elif isinstance(gain_cfg, (int, float)):
                self.gain_controller.set_manual_gain(float(gain_cfg))

        if "input_device" in settings:
            self.inference_worker.input_device = settings["input_device"]

        if "output_device" in settings:
            self.inference_worker.output_device = settings["output_device"]

        if "ptt_enabled" in settings:
            self.set_ptt_enabled(bool(settings["ptt_enabled"]))

        return {
            "status": "updated",
            "is_running": self.is_running(),
            "pitch_shift": self.rvc_engine.pitch_shift,
            "noise_suppression_enabled": self.noise_suppressor.enabled,
            "effects_enabled": self.effects_chain.enabled,
            "gain_state": self.gain_controller.get_state(),
            "ptt_enabled": self.inference_worker._ptt_enabled
        }
