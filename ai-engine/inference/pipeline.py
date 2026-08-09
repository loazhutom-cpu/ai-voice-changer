"""
Main Audio Processing Pipeline Orchestrator.

Chains noise suppression, RVC voice conversion, pedalboard effects, and low-latency audio stream routing.
Provides a unified API for starting, stopping, querying status, and updating settings.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from ai_engine.rvc.rvc_engine import RVCEngine
    from ai_engine.rvc.rvc_inference import RealtimeRVCInference
except ImportError:
    try:
        from ..rvc.rvc_engine import RVCEngine
        from ..rvc.rvc_inference import RealtimeRVCInference
    except ImportError:
        from rvc.rvc_engine import RVCEngine
        from rvc.rvc_inference import RealtimeRVCInference

from backend.audio_effects import AudioEffectsChain
from backend.noise_suppression import NoiseSuppressor

logger = logging.getLogger(__name__)


class AudioPipeline:
    """
    Main Audio Processing Pipeline Orchestrator.

    Integrates RVC Engine, Noise Suppressor, Audio Effects, and real-time audio streams.
    Runs streaming processing inside background threads.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        sample_rate: int = 48000,
        chunk_size: int = 2048,
        buffer_size: int = 8192
    ):
        """
        Initialize Audio Pipeline.

        Args:
            config_path: Path to default YAML configuration
            sample_rate: Audio sampling frequency in Hz
            chunk_size: Processing audio chunk frame size
            buffer_size: Circular audio buffer frame capacity
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size

        # Core Pipeline Components
        self.rvc_engine = RVCEngine()
        self.noise_suppressor = NoiseSuppressor()
        self.effects_chain = AudioEffectsChain()

        # Real-time Streaming Worker
        self.inference_worker = RealtimeRVCInference(
            rvc_engine=self.rvc_engine,
            noise_suppressor=self.noise_suppressor,
            effects_chain=self.effects_chain,
            sample_rate=self.sample_rate,
            chunk_size=self.chunk_size,
            buffer_size=self.buffer_size
        )

        self.current_preset: str = "default_voice"
        logger.info("AudioPipeline orchestrator initialized successfully.")

    def start(self) -> Dict[str, Any]:
        """
        Start real-time audio capture, conversion, and playback pipeline.

        Returns:
            Dict containing status and current operational parameters
        """
        logger.info("Starting AudioPipeline...")
        self.inference_worker.start()
        return {
            "status": "running",
            "active_preset": self.current_preset,
            "sample_rate": self.sample_rate,
            "chunk_size": self.chunk_size
        }

    def stop(self) -> Dict[str, Any]:
        """
        Stop real-time audio processing pipeline.

        Returns:
            Dict containing status report
        """
        logger.info("Stopping AudioPipeline...")
        self.inference_worker.stop()
        return {"status": "stopped"}

    def is_running(self) -> bool:
        """Check if conversion pipeline is currently active."""
        return self.inference_worker.is_running()

    def set_voice_preset(self, preset_id: str) -> bool:
        """
        Switch active voice model preset.

        Args:
            preset_id: Voice identifier or file path stem

        Returns:
            True if switch succeeded, False otherwise
        """
        logger.info(f"Switching voice preset to: {preset_id}")
        success = self.rvc_engine.load_model(f"ai-engine/models/presets/{preset_id}.pth")
        if success or preset_id == "default_voice":
            self.current_preset = preset_id
            return True
        return False

    def set_pitch_shift(self, semitones: float) -> None:
        """
        Set pitch transposition shift in semitones.

        Args:
            semitones: Semitone offset (-24.0 to +24.0)
        """
        self.rvc_engine.set_pitch_shift(semitones)

    def get_latency(self) -> Dict[str, float]:
        """
        Get breakdown of current latency statistics in milliseconds.

        Returns:
            Dict containing chunk_buffer_ms, processing_ms, total_latency_ms
        """
        return self.inference_worker.get_latency_stats()

    def get_audio_levels(self) -> Dict[str, float]:
        """
        Get current real-time input and output audio levels for UI meters.

        Returns:
            Dict containing RMS levels and dB readings
        """
        return self.inference_worker.get_audio_levels()

    def get_available_voices(self) -> List[Dict[str, Any]]:
        """
        List all available voice model presets.

        Returns:
            List of preset dict objects
        """
        return self.rvc_engine.get_available_voices()

    def update_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update runtime audio settings (effects, gain, devices, noise suppression).

        Args:
            settings: Dictionary of parameter updates

        Returns:
            Updated configuration dictionary
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

        if "input_device" in settings:
            self.inference_worker.input_device = settings["input_device"]

        if "output_device" in settings:
            self.inference_worker.output_device = settings["output_device"]

        return {
            "status": "updated",
            "is_running": self.is_running(),
            "pitch_shift": self.rvc_engine.pitch_shift,
            "noise_suppression_enabled": self.noise_suppressor.enabled,
            "effects_enabled": self.effects_chain.enabled
        }
