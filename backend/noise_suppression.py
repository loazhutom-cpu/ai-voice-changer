"""
Noise Suppression Module using noisereduce and RNNoise spectral gating algorithms.

Removes background ambient noise, room hums, and static before sending audio to RVC.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False

logger = logging.getLogger(__name__)


class NoiseSuppressor:
    """
    Noise Suppressor handling spectral gating and stationary/non-stationary noise removal.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Noise Suppressor.

        Args:
            config: Optional configuration dictionary
        """
        self.enabled: bool = True
        self.mode: str = "spectral_gating"  # 'spectral_gating' or 'rnnoise'
        self.prop_decrease: float = 0.85
        self.stationary: bool = True
        self.time_mask_smooth_ms: int = 50
        self.n_fft: int = 1024

        if config:
            self.update_config(config)

    def process(self, audio: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
        """
        Apply noise reduction algorithm to audio chunk.

        Args:
            audio: Input floating point audio array
            sample_rate: Sampling rate in Hz

        Returns:
            Denoised audio array of matching shape
        """
        if not self.enabled or audio is None or len(audio) == 0:
            return audio

        if HAS_NOISEREDUCE:
            try:
                # Handle multi-channel audio
                if audio.ndim > 1:
                    mono = np.mean(audio, axis=1)
                else:
                    mono = audio

                reduced = nr.reduce_noise(
                    y=mono,
                    sr=sample_rate,
                    prop_decrease=self.prop_decrease,
                    stationary=self.stationary,
                    n_fft=self.n_fft,
                    time_mask_smooth_ms=self.time_mask_smooth_ms
                )

                if audio.ndim > 1 and audio.shape[1] > 1:
                    return np.column_stack([reduced] * audio.shape[1]).astype(np.float32)
                return reduced.astype(np.float32)
            except Exception as e:
                logger.error(f"Error during noisereduce processing: {e}")
                return audio

        # Lightweight highpass noise gate fallback if noisereduce library is not present
        return self._noise_gate_fallback(audio)

    def _noise_gate_fallback(self, audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Lightweight spectral threshold noise gate fallback."""
        mask = np.abs(audio) > threshold
        return (audio * mask).astype(np.float32)

    def update_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update noise suppressor parameters.

        Args:
            config: Parameter dictionary

        Returns:
            Updated configuration dictionary
        """
        if "enabled" in config:
            self.enabled = bool(config["enabled"])
        if "prop_decrease" in config:
            self.prop_decrease = float(config["prop_decrease"])
        if "stationary" in config:
            self.stationary = bool(config["stationary"])
        if "mode" in config:
            self.mode = str(config["mode"])

        return self.get_config()

    def get_config(self) -> Dict[str, Any]:
        """Get current noise suppressor status and parameters."""
        return {
            "enabled": self.enabled,
            "has_noisereduce": HAS_NOISEREDUCE,
            "mode": self.mode,
            "prop_decrease": self.prop_decrease,
            "stationary": self.stationary
        }
