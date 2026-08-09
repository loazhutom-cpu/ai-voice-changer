"""
Voice Model Registry and Metadata Tracking Schema.

Registers available trained voice models, stores metadata (name, pitch range, sample rate),
and manages model files on disk.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class VoiceMetadata(BaseModel):
    """Voice Model Metadata Pydantic Schema."""

    id: str = Field(..., description="Unique identifier for the voice preset")
    name: str = Field(..., description="Human-readable voice model name")
    voice_type: str = Field(default="RVC v2", description="Architecture model type")
    gender: str = Field(default="neutral", description="Voice gender classification")
    sample_rate: int = Field(default=48000, description="Target sampling rate in Hz")
    pitch_range: Tuple[float, float] = Field(default=(-12.0, 12.0), description="Recommended pitch shift range")
    pitch_shift_recommended: float = Field(default=0.0, description="Default semitone pitch offset")
    description: str = Field(default="", description="Voice model description or provenance")
    model_path: str = Field(..., description="File path to .pth checkpoint file")
    index_path: Optional[str] = Field(default=None, description="File path to feature index file")
    file_size_mb: float = Field(default=0.0, description="Checkpoint file size in megabytes")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Creation timestamp")


class VoiceModelRegistry:
    """
    Registry for managing trained voice model files, scanning model directories,
    and retrieving voice metadata.
    """

    def __init__(self, presets_dir: Union[str, Path] = "ai-engine/models/presets"):
        """
        Initialize Voice Model Registry.

        Args:
            presets_dir: Directory containing .pth model files and metadata JSONs
        """
        self.presets_dir = Path(presets_dir)
        self.presets_dir.mkdir(parents=True, exist_ok=True)
        self.registry: Dict[str, VoiceMetadata] = {}
        self.scan_presets()

    def scan_presets(self) -> List[VoiceMetadata]:
        """
        Scan directory for voice model files (.pth) and associated metadata files.

        Returns:
            List of VoiceMetadata objects
        """
        self.registry.clear()

        # Step 1: Scan for .pth files
        for pth_file in self.presets_dir.glob("*.pth"):
            voice_id = pth_file.stem
            meta_file = pth_file.with_suffix(".json")
            index_file = pth_file.with_suffix(".index")

            if meta_file.exists():
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta_dict = json.load(f)
                        meta_dict["id"] = voice_id
                        meta_dict["model_path"] = str(pth_file)
                        if index_file.exists():
                            meta_dict["index_path"] = str(index_file)
                        metadata = VoiceMetadata(**meta_dict)
                except Exception as e:
                    logger.error(f"Error parsing metadata for {meta_file}: {e}")
                    metadata = self._create_default_metadata(pth_file, index_file)
            else:
                metadata = self._create_default_metadata(pth_file, index_file)

            self.registry[voice_id] = metadata

        # If empty, add standard virtual voice preset entry
        if not self.registry:
            default_meta = VoiceMetadata(
                id="default_voice",
                name="Default Studio Voice",
                voice_type="RVC v2",
                gender="neutral",
                sample_rate=48000,
                model_path=str(self.presets_dir / "default_voice.pth"),
                description="Default built-in standard voice model preset."
            )
            self.registry["default_voice"] = default_meta

        return list(self.registry.values())

    def _create_default_metadata(self, pth_file: Path, index_file: Path) -> VoiceMetadata:
        """Helper to create metadata for unindexed .pth file."""
        size_mb = round(pth_file.stat().st_size / (1024 * 1024), 2) if pth_file.exists() else 0.0
        return VoiceMetadata(
            id=pth_file.stem,
            name=pth_file.stem.replace("_", " ").title(),
            voice_type="RVC v2",
            gender="neutral",
            sample_rate=48000,
            model_path=str(pth_file),
            index_path=str(index_file) if index_file.exists() else None,
            file_size_mb=size_mb,
            description="Scanned voice checkpoint."
        )

    def get_voice(self, voice_id: str) -> Optional[VoiceMetadata]:
        """
        Get metadata for specific voice model ID.

        Args:
            voice_id: Voice identifier

        Returns:
            VoiceMetadata if found, None otherwise
        """
        return self.registry.get(voice_id)

    def register_voice(self, metadata: VoiceMetadata) -> bool:
        """
        Register a new voice model and write metadata JSON to disk.

        Args:
            metadata: VoiceMetadata instance

        Returns:
            True if registration succeeded
        """
        try:
            self.registry[metadata.id] = metadata
            meta_file = self.presets_dir / f"{metadata.id}.json"
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(metadata.model_dump(), f, indent=2)
            logger.info(f"Registered voice model '{metadata.id}' successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to register voice '{metadata.id}': {e}")
            return False

    def delete_voice(self, voice_id: str) -> bool:
        """
        Remove voice model files and unregister from registry.

        Args:
            voice_id: Voice preset identifier to delete

        Returns:
            True if deletion succeeded
        """
        if voice_id not in self.registry:
            return False

        meta = self.registry.pop(voice_id)
        pth_path = Path(meta.model_path)
        json_path = pth_path.with_suffix(".json")
        index_path = pth_path.with_suffix(".index")

        for path in [pth_path, json_path, index_path]:
            if path.exists():
                try:
                    path.unlink()
                except Exception as e:
                    logger.error(f"Error removing file {path}: {e}")

        logger.info(f"Voice preset '{voice_id}' deleted.")
        return True

    def list_voices(self) -> List[Dict[str, Any]]:
        """List all registered voices formatted as dictionary list."""
        return [meta.model_dump() for meta in self.registry.values()]
