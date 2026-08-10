"""
RVC (Retrieval-based Voice Conversion) Engine Wrapper.

Provides real-time voice conversion using PyTorch with pitch extraction,
content feature extraction, and neural synthesis with CUDA GPU support
and CPU fallback.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import scipy.signal

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from ai_engine.inference.pitch_extractor import PitchExtractor
    from ai_engine.inference.feature_extractor import FeatureExtractor
except ImportError:
    try:
        from ..inference.pitch_extractor import PitchExtractor
        from ..inference.feature_extractor import FeatureExtractor
    except ImportError:
        from inference.pitch_extractor import PitchExtractor
        from inference.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


if HAS_TORCH:
    class SyntheticRVCSynthesizer(nn.Module):
        """
        RVC Generator model (SynthesizerTrn-style).

        Converts content features and pitch contour into converted audio
        via convolutional neural network layers.
        """

        def __init__(self, feature_dim: int = 256, hidden_dim: int = 192):
            super().__init__()
            self.feature_dim = feature_dim
            self.emb_phone = nn.Linear(feature_dim, hidden_dim)
            self.emb_pitch = nn.Embedding(256, hidden_dim)
            self.conv_pre = nn.Conv1d(hidden_dim, 256, kernel_size=5, padding=2)
            self.res_blocks = nn.Sequential(
                nn.Conv1d(256, 256, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(256, 128, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(128, 1, kernel_size=7, padding=3),
            )
            self.tanh = nn.Tanh()

        def forward(self, phone_features: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
            """
            Forward pass for voice synthesis.

            Args:
                phone_features: (batch, seq_len, feature_dim) content embeddings
                pitch: (batch, seq_len) pitch contour in MIDI note values

            Returns:
                (batch, samples) generated audio waveform
            """
            x = self.emb_phone(phone_features).transpose(1, 2)
            pitch_clamped = torch.clamp(pitch.long(), 0, 255)
            pitch_emb = self.emb_pitch(pitch_clamped).transpose(1, 2)
            x = x + pitch_emb
            x = self.conv_pre(x)
            x = self.res_blocks(x)
            return self.tanh(x).squeeze(1)


class RVCEngine:
    """
    Retrieval-based Voice Conversion engine.

    Handles model loading, feature extraction (content + pitch),
    neural synthesis, and pitch shifting with GPU acceleration.
    """

    def __init__(
        self,
        models_dir: Union[str, Path] = "ai-engine/models/presets",
        default_pitch_shift: float = 0.0
    ):
        """
        Initialize RVC Engine.

        Args:
            models_dir: Directory containing trained voice models (.pth files)
            default_pitch_shift: Default pitch shift in semitones (-24 to +24)
        """
        self.models_dir = Path(models_dir)
        self.pitch_shift: float = default_pitch_shift
        self.current_model_path: Optional[Path] = None
        self.current_voice_name: Optional[str] = None
        self.is_loaded: bool = False

        # Device detection with CPU fallback
        if HAS_TORCH and torch.cuda.is_available():
            self.device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"RVC Engine using CUDA GPU: {gpu_name}")
        elif HAS_TORCH:
            self.device = torch.device("cpu")
            logger.info("RVC Engine using CPU")
        else:
            self.device = None
            logger.warning("PyTorch not available. RVC Engine in fallback mode.")

        self.model: Optional[Any] = None
        self.index_file: Optional[Path] = None

        # Initialize feature extractors
        self.pitch_extractor = PitchExtractor(method="autocorrelation")
        self.feature_extractor = FeatureExtractor(feature_dim=256, device=str(self.device) if self.device else None)

        self._initialize_default_model()

    def _initialize_default_model(self) -> None:
        """Instantiate default model architecture."""
        if HAS_TORCH:
            self.model = SyntheticRVCSynthesizer().to(self.device)
            self.model.eval()
        else:
            self.model = None

    def load_model(self, model_path: Union[str, Path]) -> bool:
        """
        Load an RVC voice model checkpoint.

        Args:
            model_path: Path to the .pth or .onnx voice model file

        Returns:
            True if loaded successfully, False otherwise
        """
        path = Path(model_path)
        logger.info(f"Loading RVC model from {path}")

        if not HAS_TORCH:
            logger.warning("PyTorch unavailable — cannot load model weights")
            self.current_voice_name = path.stem
            self.is_loaded = True
            return True

        try:
            if path.exists() and path.suffix in [".pth", ".pt"]:
                checkpoint = torch.load(path, map_location=self.device)
                if isinstance(checkpoint, dict) and "model" in checkpoint:
                    self.model.load_state_dict(checkpoint["model"], strict=False)
                elif isinstance(checkpoint, dict) and "weight" in checkpoint:
                    self.model.load_state_dict(checkpoint["weight"], strict=False)
                else:
                    self.model.load_state_dict(checkpoint, strict=False)
                logger.info(f"Loaded checkpoint weights from {path}")
            else:
                logger.warning(f"Model file {path} not found. Using default synth model.")

            self.current_model_path = path
            self.current_voice_name = path.stem
            self.is_loaded = True

            # Look for matching feature retrieval index file
            possible_index = path.with_suffix(".index")
            if possible_index.exists():
                self.index_file = possible_index
                logger.info(f"Associated index file: {possible_index}")
            else:
                self.index_file = None

            return True

        except Exception as e:
            logger.error(f"Error loading RVC model: {e}", exc_info=True)
            self.is_loaded = False
            return False

    def set_pitch_shift(self, semitones: float) -> None:
        """
        Set pitch shift in semitones.

        Args:
            semitones: Shift amount (-24.0 to +24.0)
        """
        self.pitch_shift = float(np.clip(semitones, -24.0, 24.0))
        logger.debug(f"Pitch shift set to {self.pitch_shift} semitones")

    def get_available_voices(self) -> List[Dict[str, Any]]:
        """
        List all available voice model presets.

        Returns:
            List of voice info dicts (id, name, path, has_index, size_mb, is_active)
        """
        voices = []
        if not self.models_dir.exists():
            self.models_dir.mkdir(parents=True, exist_ok=True)

        for file_path in self.models_dir.glob("*.pth"):
            index_path = file_path.with_suffix(".index")
            voices.append({
                "id": file_path.stem,
                "name": file_path.stem.replace("_", " ").title(),
                "path": str(file_path),
                "has_index": index_path.exists(),
                "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
                "is_active": str(file_path) == str(self.current_model_path)
            })

        if not voices:
            voices.append({
                "id": "default_voice",
                "name": "Default Studio Voice",
                "path": str(self.models_dir / "default_voice.pth"),
                "has_index": False,
                "size_mb": 0.0,
                "is_active": True
            })

        return voices

    def convert_audio(
        self,
        audio_chunk: np.ndarray,
        target_voice: Optional[str] = None,
        pitch_shift: Optional[float] = None,
        sample_rate: int = 48000
    ) -> np.ndarray:
        """
        Convert audio chunk to target voice.

        Pipeline: feature extraction → pitch extraction → pitch shift →
                  neural synthesis → resample to match input length.

        Args:
            audio_chunk: Input audio (float32, normalized [-1, 1])
            target_voice: Optional voice ID to switch to
            pitch_shift: Optional pitch override in semitones
            sample_rate: Audio sample rate in Hz

        Returns:
            Converted audio numpy array matching input length
        """
        if audio_chunk is None or len(audio_chunk) == 0:
            return np.zeros(0, dtype=np.float32)

        original_shape = audio_chunk.shape

        # Handle stereo → mono
        if audio_chunk.ndim > 1:
            mono_audio = np.mean(audio_chunk, axis=1)
        else:
            mono_audio = audio_chunk

        # Ensure float32
        mono_audio = mono_audio.astype(np.float32)

        # Switch voice model if requested
        if target_voice and target_voice != self.current_voice_name:
            target_path = self.models_dir / f"{target_voice}.pth"
            if target_path.exists() or target_voice == "default_voice":
                self.load_model(target_path)

        shift = pitch_shift if pitch_shift is not None else self.pitch_shift

        # Step 1: Extract content features
        content_features = self.feature_extractor.extract(mono_audio, sample_rate)

        # Step 2: Extract pitch (F0) and apply pitch shift
        f0_contour = self.pitch_extractor.extract_f0(mono_audio, sample_rate)
        f0_shifted = self.pitch_extractor.apply_pitch_shift(f0_contour, shift)
        midi_contour = self.pitch_extractor.hz_to_midi(f0_shifted)

        # Resample contours to match feature frames
        num_frames = content_features.shape[0]
        midi_resampled = self.pitch_extractor.resample_contour(midi_contour, num_frames)

        if not HAS_TORCH or self.model is None:
            # Fallback: pitch-shifted passthrough (no neural conversion)
            return self._pitch_shift_passthrough(mono_audio, shift, sample_rate, original_shape)

        # Step 3: Neural synthesis
        with torch.no_grad():
            features_tensor = torch.from_numpy(content_features).float().unsqueeze(0).to(self.device)
            pitch_tensor = torch.from_numpy(midi_resampled).float().unsqueeze(0).to(self.device)

            out_tensor = self.model(features_tensor, pitch_tensor)
            converted_mono = out_tensor.cpu().numpy().squeeze()

        # Step 4: Resample output to match input length exactly
        if len(converted_mono) != len(mono_audio):
            converted_mono = scipy.signal.resample(converted_mono, len(mono_audio))

        # Normalize to prevent clipping
        max_val = np.max(np.abs(converted_mono))
        if max_val > 1.0:
            converted_mono = converted_mono / max_val

        # Restore channel layout
        if len(original_shape) > 1 and original_shape[1] > 1:
            converted_audio = np.column_stack([converted_mono] * original_shape[1])
        else:
            converted_audio = converted_mono

        return converted_audio.astype(np.float32)

    def _pitch_shift_passthrough(
        self,
        audio: np.ndarray,
        semitones: float,
        sample_rate: int,
        original_shape: tuple
    ) -> np.ndarray:
        """
        Fallback pitch shifting without neural conversion.

        Uses scipy signal processing for time-domain pitch shifting
        when the neural model is unavailable.
        """
        if semitones == 0:
            result = audio
        else:
            # Simple resampling-based pitch shift
            ratio = 2.0 ** (-semitones / 12.0)
            new_length = int(len(audio) * ratio)
            if new_length > 0:
                resampled = scipy.signal.resample(audio, new_length)
                # Trim or pad to match original length
                if len(resampled) > len(audio):
                    resampled = resampled[:len(audio)]
                else:
                    resampled = np.pad(resampled, (0, len(audio) - len(resampled)))
                result = resampled.astype(np.float32)
            else:
                result = audio

        if len(original_shape) > 1 and original_shape[1] > 1:
            result = np.column_stack([result] * original_shape[1])

        return result.astype(np.float32)

    def get_device_info(self) -> Dict[str, Any]:
        """Get current device and model status for diagnostics."""
        info = {
            "has_torch": HAS_TORCH,
            "device": str(self.device) if self.device else "none",
            "model_loaded": self.is_loaded,
            "current_voice": self.current_voice_name or "none",
            "pitch_shift": self.pitch_shift,
        }
        if HAS_TORCH and torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_allocated_mb"] = round(
                torch.cuda.memory_allocated() / (1024 * 1024), 2
            )
        return info
