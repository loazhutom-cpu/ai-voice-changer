"""
Content Feature Extraction Module.

Extracts acoustic content features (HuBERT/ContentVec-style embeddings)
from raw audio for use in RVC voice conversion. Uses a lightweight
convolutional front-end as a stand-in for HuBERT when the full model
is unavailable.
"""

import logging
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


if HAS_TORCH:
    class LightweightContentEncoder(nn.Module):
        """
        Lightweight content feature encoder inspired by HuBERT/ContentVec.

        Uses stacked 1D convolutions to produce frame-level content embeddings
        from raw audio. This is a stand-in for the full HuBERT model —
        when a real ContentVec checkpoint is available, it should be loaded
        via the `load_hubert_model` method.
        """

        def __init__(self, input_dim: int = 1, hidden_dim: int = 256, output_dim: int = 256):
            super().__init__()
            self.output_dim = output_dim

            # Front-end feature extractor (analogous to HuBERT CNN feature extractor)
            self.conv_layers = nn.Sequential(
                nn.Conv1d(input_dim, 64, kernel_size=10, stride=5, padding=0),
                nn.GELU(),
                nn.GroupNorm(1, 64),
                nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                nn.GroupNorm(1, 128),
                nn.Conv1d(128, hidden_dim, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                nn.GroupNorm(1, hidden_dim),
            )

            # Projection to embedding dimension
            self.project = nn.Linear(hidden_dim, output_dim)

        def forward(self, audio: torch.Tensor) -> torch.Tensor:
            """
            Extract content features from audio.

            Args:
                audio: (batch, samples) raw audio tensor

            Returns:
                (batch, frames, output_dim) content embedding tensor
            """
            x = audio.unsqueeze(1)  # (batch, 1, samples)
            x = self.conv_layers(x)  # (batch, hidden_dim, frames)
            x = x.transpose(1, 2)    # (batch, frames, hidden_dim)
            x = self.project(x)      # (batch, frames, output_dim)
            return x


class FeatureExtractor:
    """
    Extracts content embeddings from audio for RVC voice conversion.

    Falls back to a lightweight encoder when the full HuBERT/ContentVec
    model is not available. Supports GPU acceleration when PyTorch+CUDA
    are installed.
    """

    def __init__(self, feature_dim: int = 256, device: Optional[str] = None):
        """
        Initialize feature extractor.

        Args:
            feature_dim: Output embedding dimension (should match RVC model)
            device: 'cuda', 'cpu', or None for auto-detection
        """
        self.feature_dim = feature_dim

        if not HAS_TORCH:
            logger.warning("PyTorch not available. Feature extraction will use numpy fallback.")
            self.device = None
            self.encoder = None
            return

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.encoder = LightweightContentEncoder(output_dim=feature_dim).to(self.device)
        self.encoder.eval()

        # Try loading a real HuBERT/ContentVec checkpoint if available
        self._hubert_model = None
        self._try_load_hubert()

        logger.info(f"FeatureExtractor initialized on device: {self.device}")

    def _try_load_hubert(self, model_path: Optional[str] = None) -> bool:
        """
        Attempt to load a pre-trained HuBERT/ContentVec model.

        Args:
            model_path: Optional path to checkpoint. If None, checks common locations.

        Returns:
            True if HuBERT model loaded successfully
        """
        try:
            # Try torch.hub for fairseq HuBERT
            if model_path is None:
                # Check if fairseq is available
                try:
                    import fairseq
                    self._hubert_model = torch.hub.load(
                        "facebookresearch/fairseq", "hubert",
                        trust_repo=True
                    ).to(self.device)
                    self._hubert_model.eval()
                    logger.info("Loaded HuBERT model via torch.hub")
                    return True
                except ImportError:
                    logger.info("fairseq not installed. Using lightweight content encoder.")
                    return False
        except Exception as e:
            logger.info(f"Could not load HuBERT model: {e}. Using lightweight encoder.")
            return False

    def extract(self, audio: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
        """
        Extract content features from an audio chunk.

        Args:
            audio: 1D float32 numpy array of audio samples
            sample_rate: Audio sample rate

        Returns:
            2D numpy array of shape (frames, feature_dim) containing content embeddings
        """
        if audio is None or len(audio) == 0:
            return np.zeros((1, self.feature_dim), dtype=np.float32)

        if not HAS_TORCH or self.encoder is None:
            return self._numpy_fallback(audio)

        # Resample to 16kHz if needed (HuBERT expects 16kHz)
        if sample_rate != 16000:
            audio = self._resample(audio, sample_rate, 16000)

        with torch.no_grad():
            tensor_audio = torch.from_numpy(audio.astype(np.float32)).to(self.device)
            if tensor_audio.dim() == 1:
                tensor_audio = tensor_audio.unsqueeze(0)  # (1, samples)

            if self._hubert_model is not None:
                # Use real HuBERT
                features = self._hubert_model.extract_features(tensor_audio)[0]
            else:
                # Use lightweight encoder
                features = self.encoder(tensor_audio)

            return features.squeeze(0).cpu().numpy()

    def _numpy_fallback(self, audio: np.ndarray) -> np.ndarray:
        """
        Simple spectral feature fallback when PyTorch is unavailable.
        Produces MFCC-like features using numpy FFT.
        """
        n_frames = max(1, len(audio) // 320)
        features = np.zeros((n_frames, self.feature_dim), dtype=np.float32)

        for i in range(n_frames):
            start = i * 320
            end = min(start + 320, len(audio))
            frame = audio[start:end]

            if len(frame) > 0:
                # Simple spectral centroid + rolloff as placeholder features
                fft = np.abs(np.fft.rfft(frame))
                if fft.sum() > 0:
                    freqs = np.arange(len(fft))
                    centroid = (freqs * fft).sum() / (fft.sum() + 1e-9)
                    features[i, 0] = float(centroid)
                    features[i, 1] = float(np.mean(fft))
                    features[i, 2] = float(np.std(fft))

        return features

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Linear interpolation resampling."""
        if orig_sr == target_sr:
            return audio
        ratio = target_sr / orig_sr
        n_out = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, n_out)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
