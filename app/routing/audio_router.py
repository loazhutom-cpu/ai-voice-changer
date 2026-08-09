"""
Audio Router Manager for AI Voice Changer.

Handles:
  - Detecting available system virtual audio devices
  - Routing processed audio to selected virtual mic outputs
  - Real-time monitoring of routing status, buffer levels, and latency
  - Output target switching for application profiles (OBS, Discord, Zoom, Teams, VRChat)
"""

import sys
import os
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

# Support relative and top-level execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../virtual-mic")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

try:
    from virtual_mic import VirtualMicrophone
except ImportError:
    # Fallback import if package structure differs
    try:
        from virtual_mic.virtual_mic import VirtualMicrophone
    except ImportError:
        VirtualMicrophone = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AudioRouter")


class TargetProfile:
    """Configuration profile for common target applications."""

    TARGETS = {
        "obs": {
            "name": "OBS Studio",
            "recommended_device": {
                "win32": "CABLE Input (VB-Audio Virtual Cable)",
                "darwin": "BlackHole 2ch",
                "linux": "AI_Voice_Changer_Sink"
            },
            "sample_rate": 48000,
            "buffer_size": 512,
            "channels": 2,
            "description": "High-fidelity stereo routing for streaming and recording in OBS Studio."
        },
        "discord": {
            "name": "Discord",
            "recommended_device": {
                "win32": "CABLE Input (VB-Audio Virtual Cable)",
                "darwin": "BlackHole 2ch",
                "linux": "AI_Voice_Changer_Sink"
            },
            "sample_rate": 48000,
            "buffer_size": 1024,
            "channels": 1,
            "description": "Low-bandwidth mono audio output optimized for Discord VoIP communication."
        },
        "zoom": {
            "name": "Zoom / Teams / Meet",
            "recommended_device": {
                "win32": "CABLE Input (VB-Audio Virtual Cable)",
                "darwin": "BlackHole 2ch",
                "linux": "AI_Voice_Changer_Sink"
            },
            "sample_rate": 44100,
            "buffer_size": 1024,
            "channels": 1,
            "description": "Standard conference call audio settings with acoustic echo cancellation support."
        },
        "vrchat": {
            "name": "VRChat / Games",
            "recommended_device": {
                "win32": "CABLE Input (VB-Audio Virtual Cable)",
                "darwin": "BlackHole 2ch",
                "linux": "AI_Voice_Changer_Sink"
            },
            "sample_rate": 48000,
            "buffer_size": 256,
            "channels": 1,
            "description": "Ultra-low latency audio output optimized for real-time spatial gaming."
        }
    }


class AudioRouter:
    """Manages audio streams, device detection, and target output profiles."""

    def __init__(self):
        self.vmic = VirtualMicrophone() if VirtualMicrophone else None
        self.active_target_key: str = "obs"
        self.current_device_name: Optional[str] = None
        self.sample_rate: int = 48000
        self.channels: int = 2
        self.buffer_size: int = 512

        # Telemetry / Status tracking
        self.total_frames_processed: int = 0
        self.dropped_frames: int = 0
        self.last_process_time: float = time.time()
        self.current_rms: float = 0.0
        self.estimated_latency_ms: float = 0.0

    def detect_virtual_devices(self) -> List[Dict[str, Any]]:
        """
        Scan system audio hardware for available virtual mic/cable devices.
        """
        virtual_devices = []
        if not HAS_SOUNDDEVICE:
            logger.warning("sounddevice not available for device detection.")
            return virtual_devices

        keywords = ["cable", "blackhole", "voicemeeter", "virtual", "null-sink", "sink", "loopback"]
        all_devices = sd.query_devices()

        for idx, dev in enumerate(all_devices):
            if dev.get("max_output_channels", 0) > 0:
                d_name = dev.get("name", "")
                is_virtual = any(kw in d_name.lower() for kw in keywords)
                virtual_devices.append({
                    "id": idx,
                    "name": d_name,
                    "channels": dev.get("max_output_channels"),
                    "default_samplerate": dev.get("default_samplerate"),
                    "is_recognized_virtual": is_virtual,
                    "hostapi": dev.get("hostapi")
                })

        return virtual_devices

    def initialize_router(
        self,
        target_app: str = "obs",
        custom_device_name: Optional[str] = None
    ) -> bool:
        """
        Configure routing parameters and create the underlying virtual mic device.
        """
        target_key = target_app.lower()
        profile = TargetProfile.TARGETS.get(target_key, TargetProfile.TARGETS["obs"])

        self.active_target_key = target_key
        self.sample_rate = profile["sample_rate"]
        self.channels = profile["channels"]
        self.buffer_size = profile["buffer_size"]

        plat = sys.platform
        recommended_dev = profile["recommended_device"].get(plat, "default")
        self.current_device_name = custom_device_name or recommended_dev

        logger.info(
            f"Initializing AudioRouter for target '{profile['name']}' "
            f"on device '{self.current_device_name}' "
            f"({self.sample_rate}Hz, {self.channels}ch, buffer: {self.buffer_size})"
        )

        if not self.vmic:
            logger.error("VirtualMicrophone instance could not be created.")
            return False

        success = self.vmic.create_virtual_device(
            device_name=self.current_device_name,
            sample_rate=self.sample_rate,
            channels=self.channels,
            blocksize=self.buffer_size
        )

        if success:
            # Estimate output latency
            self.estimated_latency_ms = (self.buffer_size / self.sample_rate) * 1000.0 * 2.0
            logger.info(f"Audio router initialized. Latency ~{self.estimated_latency_ms:.2f}ms")
            return True
        else:
            logger.error("Failed to initialize virtual device in AudioRouter.")
            return False

    def route_audio(self, audio_data: np.ndarray) -> bool:
        """
        Pass AI-transformed PCM audio chunk to virtual device and update metrics.
        """
        if not self.vmic or not self.vmic.is_active:
            self.dropped_frames += 1
            return False

        start_time = time.time()

        # Compute audio signal metrics (RMS level)
        if len(audio_data) > 0:
            self.current_rms = float(np.sqrt(np.mean(np.square(audio_data))))

        success = self.vmic.write_audio(audio_data)

        if success:
            self.total_frames_processed += len(audio_data)
            # Update measured processing time
            elapsed = time.time() - start_time
            self.last_process_time = elapsed
        else:
            self.dropped_frames += 1

        return success

    def switch_output_target(
        self,
        target_app: str,
        custom_device_name: Optional[str] = None
    ) -> bool:
        """
        Dynamically switch output target preset (e.g. switch from OBS to Discord).
        """
        logger.info(f"Switching output target to '{target_app}'...")
        if self.vmic and self.vmic.is_active:
            self.vmic.destroy_device()

        return self.initialize_router(target_app=target_app, custom_device_name=custom_device_name)

    def get_routing_status(self) -> Dict[str, Any]:
        """
        Returns real-time status summary of the audio routing engine.
        """
        target_profile = TargetProfile.TARGETS.get(self.active_target_key, {})
        return {
            "active": self.vmic.is_active if self.vmic else False,
            "target_app": target_profile.get("name", self.active_target_key),
            "target_key": self.active_target_key,
            "device_name": self.current_device_name,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "buffer_size": self.buffer_size,
            "estimated_latency_ms": round(self.estimated_latency_ms, 2),
            "current_rms": round(self.current_rms, 4),
            "rms_db": round(20 * np.log10(max(self.current_rms, 1e-5)), 1),
            "total_frames_processed": self.total_frames_processed,
            "dropped_frames": self.dropped_frames,
            "platform": sys.platform
        }

    def close(self):
        """Clean up and release audio output resources."""
        if self.vmic:
            self.vmic.destroy_device()
        logger.info("AudioRouter shut down successfully.")


if __name__ == "__main__":
    print("Testing AudioRouter...")
    router = AudioRouter()
    devices = router.detect_virtual_devices()
    print(f"Detected {len(devices)} potential virtual audio devices.")

    if router.initialize_router("discord"):
        print("Status:", router.get_routing_status())
        # Simulate 10 frames of audio
        dummy_chunk = (0.1 * np.random.randn(512)).astype(np.float32)
        for _ in range(10):
            router.route_audio(dummy_chunk)
            time.sleep(512 / 48000)

        print("Updated Status:", router.get_routing_status())
        router.close()
