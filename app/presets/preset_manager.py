"""
Voice Preset Manager for AI Voice Changer.

Manages loading, creating, updating, deleting, and exporting voice preset configurations.
Handles preset persistent storage in JSON files.
"""

import os
import json
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PresetManager")


class PresetManager:
    """CRUD manager for AI Voice Changer preset profiles."""

    def __init__(self, storage_dir: Optional[str] = None, defaults_path: Optional[str] = None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.storage_dir = storage_dir or os.path.join(base_dir, "user_presets")
        self.defaults_path = defaults_path or os.path.join(base_dir, "default_presets.json")

        os.makedirs(self.storage_dir, exist_ok=True)
        self.presets: Dict[str, Dict[str, Any]] = {}
        self.load_all_presets()

    def _get_iso_timestamp(self) -> str:
        """Return current ISO formatted timestamp."""
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    def load_default_presets(self) -> List[Dict[str, Any]]:
        """Load default built-in presets from default_presets.json."""
        if not os.path.exists(self.defaults_path):
            logger.warning(f"Default presets file not found at {self.defaults_path}")
            return []

        try:
            with open(self.defaults_path, "r", encoding="utf-8") as f:
                defaults = json.load(f)
                for preset in defaults:
                    pid = preset.get("id")
                    if pid:
                        self.presets[pid] = preset
                logger.info(f"Loaded {len(defaults)} default presets.")
                return defaults
        except Exception as e:
            logger.error(f"Failed to parse default presets JSON: {e}")
            return []

    def load_all_presets(self):
        """Load both default presets and user presets from storage directory."""
        self.presets.clear()
        # First load defaults
        self.load_default_presets()

        # Load user presets from storage_dir
        if os.path.exists(self.storage_dir):
            for filename in os.listdir(self.storage_dir):
                if filename.endswith(".json"):
                    filepath = os.path.join(self.storage_dir, filename)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            preset_data = json.load(f)
                            pid = preset_data.get("id")
                            if pid:
                                self.presets[pid] = preset_data
                    except Exception as e:
                        logger.error(f"Error reading preset file {filename}: {e}")

        logger.info(f"Total presets active in manager: {len(self.presets)}")

    def list_presets(self) -> List[Dict[str, Any]]:
        """Return list of all registered presets."""
        return list(self.presets.values())

    def get_preset(self, preset_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a preset by its ID."""
        return self.presets.get(preset_id)

    def create_preset(
        self,
        name: str,
        voice_model: str,
        pitch_shift: float = 0.0,
        effects_config: Optional[Dict[str, Any]] = None,
        description: str = "",
        preset_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new voice preset and persist to disk.
        """
        now = self._get_iso_timestamp()
        pid = preset_id or f"user-preset-{uuid.uuid4().hex[:8]}"

        preset_data: Dict[str, Any] = {
            "id": pid,
            "name": name,
            "voice_model": voice_model,
            "pitch_shift": float(pitch_shift),
            "effects_config": effects_config or {
                "reverb": {"enabled": False},
                "compressor": {"enabled": False},
                "equalizer": {"enabled": False},
                "formant_shift": 0.0,
                "noise_gate": {"enabled": True, "threshold_db": -40.0}
            },
            "description": description,
            "created_at": now,
            "updated_at": now
        }

        self.presets[pid] = preset_data
        self._save_preset_to_disk(preset_data)
        logger.info(f"Created new preset '{name}' (ID: {pid})")
        return preset_data

    def update_preset(self, preset_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update an existing preset's properties.
        """
        if preset_id not in self.presets:
            logger.error(f"Cannot update: Preset '{preset_id}' not found.")
            return None

        preset = self.presets[preset_id]
        for key, val in updates.items():
            if key not in ["id", "created_at"]:
                preset[key] = val

        preset["updated_at"] = self._get_iso_timestamp()
        self.presets[preset_id] = preset
        self._save_preset_to_disk(preset)
        logger.info(f"Updated preset '{preset_id}'")
        return preset

    def delete_preset(self, preset_id: str) -> bool:
        """
        Delete a preset from memory and persistent disk storage.
        """
        if preset_id not in self.presets:
            logger.error(f"Cannot delete: Preset '{preset_id}' not found.")
            return False

        del self.presets[preset_id]

        filepath = os.path.join(self.storage_dir, f"{preset_id}.json")
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.info(f"Deleted preset file {filepath}")
            except Exception as e:
                logger.error(f"Failed to delete preset file {filepath}: {e}")

        return True

    def _save_preset_to_disk(self, preset_data: Dict[str, Any]):
        """Helper to serialize preset object to JSON file in storage directory."""
        filename = f"{preset_data['id']}.json"
        filepath = os.path.join(self.storage_dir, filename)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(preset_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save preset JSON to {filepath}: {e}")

    def export_preset(self, preset_id: str, export_path: str) -> bool:
        """Export preset object to specified file path."""
        preset = self.get_preset(preset_id)
        if not preset:
            return False
        try:
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(preset, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to export preset {preset_id}: {e}")
            return False

    def import_preset(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Import preset from external JSON file."""
        if not os.path.exists(file_path):
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                preset_data = json.load(f)
                pid = preset_data.get("id") or f"imported-{uuid.uuid4().hex[:8]}"
                preset_data["id"] = pid
                preset_data["updated_at"] = self._get_iso_timestamp()
                self.presets[pid] = preset_data
                self._save_preset_to_disk(preset_data)
                return preset_data
        except Exception as e:
            logger.error(f"Failed to import preset from {file_path}: {e}")
            return None


if __name__ == "__main__":
    print("Testing PresetManager...")
    pm = PresetManager()
    presets = pm.list_presets()
    print(f"Loaded {len(presets)} presets:")
    for p in presets:
        print(f" - [{p['id']}] {p['name']} (Model: {p['voice_model']}, Pitch: {p['pitch_shift']})")

    # Test Create & Delete
    new_p = pm.create_preset("Test Echo Voice", "echo_v1.pth", pitch_shift=2.0, description="Test description")
    print("Created:", new_p["id"])
    pm.delete_preset(new_p["id"])
    print("Test complete.")
