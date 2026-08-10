"""
Automatic Gain Control (AGC) and Manual Gain Module.

Provides real-time input gain staging with automatic level normalization
to maintain consistent audio levels before and after voice conversion.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class GainController:
    """
    Real-time gain controller with automatic gain control (AGC) support.

    Modes:
    - 'manual': Fixed gain multiplier applied to all audio
    - 'agc': Automatic gain control targeting a specific RMS level
    - 'bypass': No gain adjustment (passthrough)
    """

    def __init__(
        self,
        mode: str = "manual",
        manual_gain_db: float = 0.0,
        agc_target_rms: float = 0.15,
        agc_attack_ms: float = 20.0,
        agc_release_ms: float = 300.0,
        max_gain_db: float = 30.0,
        min_gain_db: float = -30.0
    ):
        """
        Initialize gain controller.

        Args:
            mode: 'manual', 'agc', or 'bypass'
            manual_gain_db: Fixed gain in dB for manual mode
            agc_target_rms: Target RMS level (0-1) for AGC mode
            agc_attack_ms: AGC attack time (how fast to increase gain)
            agc_release_ms: AGC release time (how fast to decrease gain)
            max_gain_db: Maximum gain ceiling in dB
            min_gain_db: Minimum gain floor in dB
        """
        self.mode = mode
        self.manual_gain_db = manual_gain_db
        self.agc_target_rms = agc_target_rms
        self.max_gain_db = max_gain_db
        self.min_gain_db = min_gain_db

        # AGC state
        self._current_gain_db = 0.0
        self._agc_attack_coeff = 0.0
        self._agc_release_coeff = 0.0
        self._set_agc_coefficients(agc_attack_ms, agc_release_ms)

        # Noise gate
        self.noise_gate_db = -60.0
        self.noise_gate_enabled = True

    def _set_agc_coefficients(self, attack_ms: float, release_ms: float, sample_rate: int = 48000) -> None:
        """Calculate AGC smoothing coefficients from time constants."""
        block_ms = (2048 / sample_rate) * 1000  # Approximate block duration
        self._agc_attack_coeff = 1.0 - np.exp(-block_ms / max(attack_ms, 0.1))
        self._agc_release_coeff = 1.0 - np.exp(-block_ms / max(release_ms, 0.1))

    def process(self, audio: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
        """
        Apply gain to audio chunk.

        Args:
            audio: 1D or 2D float32 numpy array
            sample_rate: Sample rate for AGC timing

        Returns:
            Gain-adjusted audio array
        """
        if audio is None or len(audio) == 0:
            return audio

        if self.mode == "bypass":
            return audio

        # Noise gate: silence very quiet signals
        if self.noise_gate_enabled:
            rms = self._compute_rms(audio)
            if rms > 0:
                rms_db = 20 * np.log10(rms + 1e-9)
                if rms_db < self.noise_gate_db:
                    return np.zeros_like(audio)

        if self.mode == "manual":
            gain_linear = 10 ** (self.manual_gain_db / 20.0)
            return np.clip(audio * gain_linear, -1.0, 1.0).astype(np.float32)

        elif self.mode == "agc":
            return self._apply_agc(audio)

        return audio

    def _apply_agc(self, audio: np.ndarray) -> np.ndarray:
        """Apply automatic gain control with smooth gain ramping."""
        rms = self._compute_rms(audio)

        if rms < 1e-6:
            # Signal too quiet — don't boost noise
            return audio.astype(np.float32)

        # Calculate desired gain
        desired_gain_linear = self.agc_target_rms / (rms + 1e-9)
        desired_gain_db = 20 * np.log10(desired_gain_linear + 1e-9)

        # Clamp to range
        desired_gain_db = np.clip(desired_gain_db, self.min_gain_db, self.max_gain_db)

        # Smooth gain transition (attack for increasing, release for decreasing)
        if desired_gain_db > self._current_gain_db:
            coeff = self._agc_attack_coeff
        else:
            coeff = self._agc_release_coeff

        self._current_gain_db += coeff * (desired_gain_db - self._current_gain_db)

        gain_linear = 10 ** (self._current_gain_db / 20.0)
        return np.clip(audio * gain_linear, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _compute_rms(audio: np.ndarray) -> float:
        """Compute RMS level of audio."""
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        return float(np.sqrt(np.mean(audio ** 2) + 1e-9))

    def set_mode(self, mode: str) -> None:
        """Set gain mode: 'manual', 'agc', or 'bypass'."""
        if mode in ("manual", "agc", "bypass"):
            self.mode = mode
            logger.info(f"Gain mode set to: {mode}")
        else:
            logger.warning(f"Invalid gain mode: {mode}")

    def set_manual_gain(self, gain_db: float) -> None:
        """Set manual gain in dB (-30 to +30)."""
        self.manual_gain_db = float(np.clip(gain_db, -30.0, 30.0))

    def get_state(self) -> dict:
        """Get current gain controller state for UI display."""
        return {
            "mode": self.mode,
            "manual_gain_db": self.manual_gain_db,
            "agc_target_rms": self.agc_target_rms,
            "current_gain_db": round(self._current_gain_db, 2),
            "noise_gate_enabled": self.noise_gate_enabled,
            "noise_gate_db": self.noise_gate_db
        }
