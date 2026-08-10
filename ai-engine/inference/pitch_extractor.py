"""
Pitch (F0) Extraction Module.

Provides fundamental frequency estimation using autocorrelation and
pyin-based methods for real-time voice conversion pitch tracking.
"""

import logging
from typing import Optional

import numpy as np

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

logger = logging.getLogger(__name__)


class PitchExtractor:
    """
    Extracts fundamental frequency (F0) contour from speech audio.

    Supports autocorrelation (fast, CPU) and pyin (accurate, librosa) methods.
    Returns pitch in Hz, which is then converted to semitone-based MIDI notes
    for the RVC synthesizer.
    """

    # MIDI note 69 = A4 = 440 Hz
    A4_MIDI = 69
    A4_FREQ = 440.0

    def __init__(self, method: str = "autocorrelation", frame_length: int = 2048,
                 hop_length: int = 512, fmin: float = 65.0, fmax: float = 1000.0):
        """
        Initialize pitch extractor.

        Args:
            method: 'autocorrelation' or 'pyin'
            frame_length: Analysis frame size in samples
            hop_length: Hop size between frames in samples
            fmin: Minimum detectable frequency (Hz) — male ~65Hz
            fmax: Maximum detectable frequency (Hz) — female ~1000Hz
        """
        self.method = method
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax

    def extract_f0(self, audio: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
        """
        Extract F0 contour from audio.

        Args:
            audio: 1D float32 audio array
            sample_rate: Sample rate in Hz

        Returns:
            1D numpy array of F0 values in Hz (0.0 = unvoiced)
        """
        if audio is None or len(audio) == 0:
            return np.zeros(0, dtype=np.float32)

        if self.method == "pyin" and HAS_LIBROSA:
            return self._extract_pyin(audio, sample_rate)
        else:
            return self._extract_autocorrelation(audio, sample_rate)

    def _extract_pyin(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Use librosa.pyin for high-accuracy F0 tracking."""
        try:
            f0, voiced_flag, _ = librosa.pyin(
                audio.astype(np.float64),
                fmin=self.fmin,
                fmax=self.fmax,
                sr=sample_rate,
                frame_length=self.frame_length,
                hop_length=self.hop_length
            )
            f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
            return f0
        except Exception as e:
            logger.warning(f"pyin extraction failed ({e}), falling back to autocorrelation")
            return self._extract_autocorrelation(audio, sample_rate)

    def _extract_autocorrelation(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Fast autocorrelation-based F0 estimation.

        Works well for real-time use — O(N*logN) via FFT.
        """
        audio = audio.astype(np.float64)
        n = len(audio)
        if n < 2:
            return np.zeros(0, dtype=np.float32)

        # Compute autocorrelation via FFT
        fft_size = 1
        while fft_size < 2 * n:
            fft_size *= 2

        fft = np.fft.rfft(audio, fft_size)
        autocorr = np.fft.irfft(fft * np.conj(fft), fft_size)[:n]

        # Normalize
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]

        # Search range in samples
        min_lag = int(sample_rate / self.fmax)
        max_lag = int(sample_rate / self.fmin)
        max_lag = min(max_lag, n - 1)

        if max_lag <= min_lag:
            return np.array([0.0], dtype=np.float32)

        # Find peak in autocorrelation within valid lag range
        search_region = autocorr[min_lag:max_lag + 1]
        peak_idx = np.argmax(search_region)
        peak_value = search_region[peak_idx]

        # Threshold for voiced detection
        if peak_value < 0.3:
            return np.array([0.0], dtype=np.float32)

        # Parabolic interpolation for sub-sample accuracy
        lag = min_lag + peak_idx
        if 0 < lag < n - 1:
            alpha = autocorr[lag - 1]
            beta = autocorr[lag]
            gamma = autocorr[lag + 1]
            denom = alpha - 2 * beta + gamma
            if denom != 0:
                lag_refined = lag + 0.5 * (alpha - gamma) / denom
            else:
                lag_refined = float(lag)
        else:
            lag_refined = float(lag)

        if lag_refined > 0:
            f0 = float(sample_rate / lag_refined)
        else:
            f0 = 0.0

        return np.array([f0], dtype=np.float32)

    def hz_to_midi(self, f0_hz: np.ndarray) -> np.ndarray:
        """
        Convert frequency in Hz to MIDI note numbers.

        Args:
            f0_hz: Array of frequencies (0 = unvoiced)

        Returns:
            Array of MIDI note floats (0 = unvoiced)
        """
        midi = np.zeros_like(f0_hz, dtype=np.float32)
        voiced = f0_hz > 0
        midi[voiced] = self.A4_MIDI + 12 * np.log2(f0_hz[voiced] / self.A4_FREQ)
        return midi

    def apply_pitch_shift(self, f0_hz: np.ndarray, semitones: float) -> np.ndarray:
        """
        Shift F0 contour by a number of semitones.

        Args:
            f0_hz: F0 contour in Hz
            semitones: Shift amount (-24 to +24)

        Returns:
            Shifted F0 contour in Hz
        """
        if semitones == 0:
            return f0_hz

        shifted = np.zeros_like(f0_hz, dtype=np.float32)
        voiced = f0_hz > 0
        shifted[voiced] = f0_hz[voiced] * (2.0 ** (semitones / 12.0))
        return shifted

    def resample_contour(self, contour: np.ndarray, target_length: int) -> np.ndarray:
        """
        Resample F0 contour to match a target frame count using linear interpolation.

        Args:
            contour: Input contour array
            target_length: Desired output length

        Returns:
            Resampled contour of length target_length
        """
        if len(contour) == 0:
            return np.zeros(target_length, dtype=np.float32)
        if len(contour) == target_length:
            return contour.astype(np.float32)

        indices = np.linspace(0, len(contour) - 1, target_length)
        return np.interp(indices, np.arange(len(contour)), contour).astype(np.float32)
