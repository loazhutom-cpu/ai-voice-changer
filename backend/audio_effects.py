"""
Audio Effects Chain Module using Pedalboard.

Provides real-time dynamic range compression, de-essing, equalizing, reverb, and limiting
for output vocal processing.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

try:
    from pedalboard import Chorus, Compressor, Delay, HighpassFilter, HighShelfFilter, Limiter, LowShelfFilter, PeakFilter, Pedalboard, Reverb
    HAS_PEDALBOARD = True
except ImportError:
    HAS_PEDALBOARD = False

logger = logging.getLogger(__name__)


class AudioEffectsChain:
    """
    Audio Effects Processor managing Compressor, De-Esser, Reverb, Parametric EQ, and Limiter.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Audio Effects Chain.

        Args:
            config: Optional initial configuration dictionary
        """
        self.enabled: bool = True
        self.config: Dict[str, Any] = {
            "compressor": {
                "enabled": True,
                "threshold_db": -16.0,
                "ratio": 3.0,
                "attack_ms": 10.0,
                "release_ms": 100.0
            },
            "de_esser": {
                "enabled": True,
                "threshold_db": -20.0,
                "frequency_hz": 6000.0
            },
            "reverb": {
                "enabled": False,
                "room_size": 0.25,
                "wet_level": 0.15,
                "dry_level": 0.85,
                "damping": 0.5
            },
            "eq": {
                "enabled": True,
                "low_gain_db": 0.0,
                "mid_gain_db": 1.5,
                "high_gain_db": 2.0
            },
            "limiter": {
                "enabled": True,
                "threshold_db": -1.0,
                "release_ms": 50.0
            }
        }

        if config:
            self.update_config(config)

        self._board: Optional[Any] = None
        self._rebuild_board()

    def _rebuild_board(self) -> None:
        """Reconstruct Pedalboard pipeline graph from configuration state."""
        if not HAS_PEDALBOARD:
            logger.warning("Pedalboard library not installed. Falling back to clean software DSP pass-through.")
            return

        effects = []

        # 1. Compressor
        comp_cfg = self.config.get("compressor", {})
        if comp_cfg.get("enabled", True):
            effects.append(
                Compressor(
                    threshold_db=comp_cfg.get("threshold_db", -16.0),
                    ratio=comp_cfg.get("ratio", 3.0),
                    attack_ms=comp_cfg.get("attack_ms", 10.0),
                    release_ms=comp_cfg.get("release_ms", 100.0)
                )
            )

        # 2. De-Esser (HighShelf attenuation filter)
        deess_cfg = self.config.get("de_esser", {})
        if deess_cfg.get("enabled", True):
            effects.append(
                HighShelfFilter(
                    cutoff_frequency_hz=deess_cfg.get("frequency_hz", 6000.0),
                    gain_db=-3.0
                )
            )

        # 3. Parametric EQ (Low, Mid, High bands)
        eq_cfg = self.config.get("eq", {})
        if eq_cfg.get("enabled", True):
            effects.append(LowShelfFilter(cutoff_frequency_hz=200.0, gain_db=eq_cfg.get("low_gain_db", 0.0)))
            effects.append(PeakFilter(cutoff_frequency_hz=2500.0, gain_db=eq_cfg.get("mid_gain_db", 1.5), q=1.0))
            effects.append(HighShelfFilter(cutoff_frequency_hz=8000.0, gain_db=eq_cfg.get("high_gain_db", 2.0)))

        # 4. Reverb
        rev_cfg = self.config.get("reverb", {})
        if rev_cfg.get("enabled", False):
            effects.append(
                Reverb(
                    room_size=rev_cfg.get("room_size", 0.25),
                    wet_level=rev_cfg.get("wet_level", 0.15),
                    dry_level=rev_cfg.get("dry_level", 0.85),
                    damping=rev_cfg.get("damping", 0.5)
                )
            )

        # 5. Brickwall Limiter
        lim_cfg = self.config.get("limiter", {})
        if lim_cfg.get("enabled", True):
            effects.append(
                Limiter(
                    threshold_db=lim_cfg.get("threshold_db", -1.0)
                )
            )

        self._board = Pedalboard(effects)

    def process(self, audio: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
        """
        Apply pedalboard effects chain to audio numpy array.

        Args:
            audio: Input audio NumPy array (1D or 2D)
            sample_rate: Audio sample rate in Hz

        Returns:
            Processed audio NumPy array of matching shape
        """
        if not self.enabled or audio is None or len(audio) == 0:
            return audio

        if HAS_PEDALBOARD and self._board is not None:
            try:
                # Ensure float32 dtype
                audio_f32 = audio.astype(np.float32)
                processed = self._board(audio_f32, sample_rate=sample_rate)
                return processed
            except Exception as e:
                logger.error(f"Error applying pedalboard effects: {e}")
                return audio

        # Fallback DSP soft limiter if pedalboard is unavailable
        return self._software_limiter(audio)

    def _software_limiter(self, audio: np.ndarray, threshold: float = 0.95) -> np.ndarray:
        """Simple soft-knee peak limiter fallback."""
        return np.clip(audio, -threshold, threshold).astype(np.float32)

    def update_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update effects configuration and rebuild pedalboard graph.

        Args:
            config: Parameter key-value dictionary

        Returns:
            Updated full configuration dictionary
        """
        if "enabled" in config:
            self.enabled = bool(config["enabled"])

        for key in ["compressor", "de_esser", "reverb", "eq", "limiter"]:
            if key in config and isinstance(config[key], dict):
                self.config[key].update(config[key])

        self._rebuild_board()
        return self.get_config()

    def get_config(self) -> Dict[str, Any]:
        """Get current effects configuration."""
        return {
            "enabled": self.enabled,
            "has_pedalboard": HAS_PEDALBOARD,
            "effects": self.config
        }
