"""
Virtual Microphone Output Module for AI Voice Changer.

Provides a cross-platform unified interface to interface with virtual audio devices:
  - Windows: VB-CABLE / Voicemeeter via WASAPI / DirectSound
  - macOS: BlackHole 2ch/16ch via CoreAudio
  - Linux: PulseAudio / PipeWire module-null-sink virtual device

Unified API:
  - create_virtual_device(device_name=None, sample_rate=44100, channels=1, blocksize=1024)
  - write_audio(chunk)
  - destroy_device()
"""

import sys
import os
import platform
import subprocess
import logging
import time
import numpy as np
from typing import Optional, Union, List, Dict, Any

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VirtualMic")


class VirtualMicrophone:
    """Cross-platform virtual microphone audio sink router."""

    DEFAULT_DEVICES = {
        "win32": ["CABLE Input (VB-Audio Virtual Cable)", "VoiceMeeter Input", "VB-Audio"],
        "darwin": ["BlackHole 2ch", "BlackHole 16ch", "Existual Audio"],
        "linux": ["AI_Voice_Changer_Sink", "pulse", "pipewire", "default"]
    }

    def __init__(self):
        self.platform = sys.platform
        self.device_name: Optional[str] = None
        self.device_index: Optional[int] = None
        self.sample_rate: int = 44100
        self.channels: int = 1
        self.blocksize: int = 1024
        self.stream: Optional[Any] = None
        self.is_active: bool = False
        self._linux_module_id: Optional[str] = None

    def get_platform_info(self) -> Dict[str, str]:
        """Detect underlying OS and audio architecture."""
        return {
            "os": self.platform,
            "system": platform.system(),
            "release": platform.release(),
            "architecture": platform.machine()
        }

    def setup_platform_sink(self, custom_name: Optional[str] = None) -> str:
        """Ensure virtual audio sink exists at the system level."""
        target_name = custom_name

        if self.platform == "linux":
            target_name = target_name or "AI_Voice_Changer_Sink"
            logger.info(f"Setting up Linux PulseAudio/PipeWire virtual sink: {target_name}")
            try:
                # Check if sink already exists
                result = subprocess.run(
                    ["pactl", "list", "sinks", "short"],
                    capture_output=True, text=True, check=False
                )
                if target_name not in result.stdout:
                    # Create null-sink
                    cmd = [
                        "pactl", "load-module", "module-null-sink",
                        f"sink_name={target_name}",
                        f"sink_properties=device.description={target_name}"
                    ]
                    mod_proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    self._linux_module_id = mod_proc.stdout.strip()
                    logger.info(f"Loaded Linux virtual sink module ID: {self._linux_module_id}")
                else:
                    logger.info(f"Linux virtual sink '{target_name}' already present.")
            except Exception as e:
                logger.warning(f"Could not automatically create Linux null-sink: {e}")

        elif self.platform == "darwin":
            logger.info("macOS detected. Please ensure 'BlackHole' is installed (`brew install blackhole-2ch`).")
            target_name = target_name or "BlackHole 2ch"

        elif self.platform == "win32":
            logger.info("Windows detected. Please ensure 'VB-CABLE Virtual Audio Device' is installed.")
            target_name = target_name or "CABLE Input (VB-Audio Virtual Cable)"

        return target_name

    def find_device_index(self, target_name: str) -> Optional[int]:
        """Locate device index in sounddevice matching target name."""
        if not HAS_SOUNDDEVICE:
            logger.error("sounddevice package is not installed.")
            return None

        devices = sd.query_devices()
        logger.debug(f"Available audio output devices:\n{devices}")

        # Exact or substring match
        for idx, dev in enumerate(devices):
            # Focus on output channels > 0
            if dev.get('max_output_channels', 0) > 0:
                name = dev.get('name', '')
                if target_name.lower() in name.lower():
                    logger.info(f"Matched device '{name}' at index {idx}")
                    return idx

        # Fallback search for default system names
        defaults = self.DEFAULT_DEVICES.get(self.platform, [])
        for fallback in defaults:
            for idx, dev in enumerate(devices):
                if dev.get('max_output_channels', 0) > 0 and fallback.lower() in dev.get('name', '').lower():
                    logger.info(f"Fallback matched device '{dev['name']}' at index {idx}")
                    return idx

        logger.warning(f"Could not find matching device for '{target_name}'. Using system default output.")
        return None

    def create_virtual_device(
        self,
        device_name: Optional[str] = None,
        sample_rate: int = 44100,
        channels: int = 1,
        blocksize: int = 1024
    ) -> bool:
        """
        Initialize and open the virtual audio output stream.
        """
        if self.is_active:
            logger.warning("Virtual mic already running. Destroying existing instance before re-creating.")
            self.destroy_device()

        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize

        resolved_name = self.setup_platform_sink(device_name)
        self.device_name = resolved_name

        if not HAS_SOUNDDEVICE:
            logger.error("Cannot create audio stream: sounddevice is not available.")
            return False

        self.device_index = self.find_device_index(resolved_name)

        try:
            self.stream = sd.OutputStream(
                device=self.device_index,
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.blocksize,
                dtype='float32'
            )
            self.stream.start()
            self.is_active = True
            logger.info(
                f"Virtual mic active on device '{resolved_name}' "
                f"(index: {self.device_index}, sr: {sample_rate}, ch: {channels})"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to open virtual mic OutputStream: {e}")
            self.is_active = False
            return False

    def write_audio(self, chunk: Union[np.ndarray, bytes]) -> bool:
        """
        Write PCM audio chunk to the virtual microphone stream.
        `chunk` can be float32 numpy array or raw byte buffer.
        """
        if not self.is_active or self.stream is None:
            logger.error("Virtual mic stream is not active.")
            return False

        try:
            if isinstance(chunk, bytes):
                # Convert raw PCM int16/float32 bytes to float32 numpy array
                data = np.frombuffer(chunk, dtype=np.float32)
            else:
                data = np.asarray(chunk, dtype=np.float32)

            # Ensure proper channel dimensions
            if data.ndim == 1 and self.channels > 1:
                data = np.column_stack([data] * self.channels)
            elif data.ndim == 1 and self.channels == 1:
                data = np.reshape(data, (-1, 1))

            self.stream.write(data)
            return True
        except Exception as e:
            logger.error(f"Error writing audio chunk to virtual device: {e}")
            return False

    def destroy_device(self):
        """Stop audio stream and unload virtual sinks if created dynamically."""
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
                logger.info("Virtual mic stream stopped.")
            except Exception as e:
                logger.error(f"Error stopping audio stream: {e}")
            finally:
                self.stream = None

        self.is_active = False

        if self.platform == "linux" and self._linux_module_id:
            try:
                subprocess.run(
                    ["pactl", "unload-module", self._linux_module_id],
                    capture_output=True, text=True, check=False
                )
                logger.info(f"Unloaded Linux null-sink module {self._linux_module_id}")
            except Exception as e:
                logger.warning(f"Failed to unload Linux virtual sink module: {e}")
            self._linux_module_id = None


# Module-level singleton instance for direct function imports
_default_virtual_mic = VirtualMicrophone()


def create_virtual_device(
    device_name: Optional[str] = None,
    sample_rate: int = 44100,
    channels: int = 1,
    blocksize: int = 1024
) -> bool:
    """Module function wrapper to create virtual device."""
    return _default_virtual_mic.create_virtual_device(
        device_name=device_name,
        sample_rate=sample_rate,
        channels=channels,
        blocksize=blocksize
    )


def write_audio(chunk: Union[np.ndarray, bytes]) -> bool:
    """Module function wrapper to write audio chunk."""
    return _default_virtual_mic.write_audio(chunk)


def destroy_device():
    """Module function wrapper to destroy virtual device."""
    _default_virtual_mic.destroy_device()


if __name__ == "__main__":
    print("Testing Virtual Microphone Module...")
    vmic = VirtualMicrophone()
    print("Platform Info:", vmic.get_platform_info())
    success = create_virtual_device(sample_rate=44100, channels=1, blocksize=1024)
    if success:
        print("Generating test tone for 1 second...")
        t = np.linspace(0, 1, 44100, endpoint=False)
        tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        # Write in chunks
        chunk_size = 1024
        for i in range(0, len(tone), chunk_size):
            write_audio(tone[i:i+chunk_size])
            time.sleep(chunk_size / 44100)
        destroy_device()
        print("Test complete.")
