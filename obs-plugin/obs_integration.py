"""
OBS Studio Integration Module for AI Voice Changer.

Provides high-level controls for OBS Studio via OBS WebSocket API v5:
  - connect() with automatic reconnection logic
  - set_audio_input_source(device_name)
  - enable_monitoring() / disable_monitoring()
  - get_audio_monitoring_state()
  - set_volume(source_name, volume_db)
"""

import os
import json
import time
import logging
import threading
from typing import Optional, Dict, Any, Union

from obs_websocket_client import OBSWebSocketClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OBSIntegration")


class OBSController:
    """High-level controller for OBS Studio integration."""

    MONITORING_TYPES = {
        "NONE": "OBS_MONITORING_TYPE_NONE",
        "MONITOR_ONLY": "OBS_MONITORING_TYPE_MONITOR_ONLY",
        "MONITOR_AND_OUTPUT": "OBS_MONITORING_TYPE_MONITOR_AND_OUTPUT"
    }

    def __init__(self, host: str = "localhost", port: int = 4455, password: str = "", default_source: str = "Mic/Aux"):
        self.host = host
        self.port = port
        self.password = password
        self.source_name = default_source

        self.client = OBSWebSocketClient(host=self.host, port=self.port, password=self.password)
        self.auto_reconnect = True
        self.reconnect_interval = 5.0
        self._reconnect_thread: Optional[threading.Thread] = None
        self._stop_reconnect = threading.Event()

    def connect(self) -> bool:
        """Connect to OBS Studio and start background reconnection watcher."""
        success = self.client.connect()
        if success:
            logger.info(f"Connected to OBS Studio at {self.host}:{self.port}")
            self._start_reconnect_monitor()
        else:
            logger.warning("Initial OBS connection failed. Background reconnect will retry.")
            self._start_reconnect_monitor()
        return success

    def disconnect(self):
        """Disconnect from OBS Studio and stop reconnect loop."""
        self._stop_reconnect.set()
        if self._reconnect_thread and self._reconnect_thread.is_alive():
            self._reconnect_thread.join(timeout=2.0)
        self.client.disconnect()

    def _start_reconnect_monitor(self):
        """Launch reconnect monitoring thread if not already running."""
        if self._reconnect_thread and self._reconnect_thread.is_alive():
            return
        self._stop_reconnect.clear()
        self._reconnect_thread = threading.Thread(target=self._reconnect_loop, daemon=True)
        self._reconnect_thread.start()

    def _reconnect_loop(self):
        """Continuously verify connection and reconnect if dropped."""
        while not self._stop_reconnect.is_set():
            time.sleep(self.reconnect_interval)
            if not self.client.is_connected or not self.client.is_authenticated:
                logger.info("OBS connection offline. Attempting reconnect...")
                res = self.client.connect()
                if res:
                    logger.info("Reconnected to OBS Studio!")

    def is_connected(self) -> bool:
        """Return True if OBS WebSocket client is active and authenticated."""
        return self.client.is_connected and self.client.is_authenticated

    def set_audio_input_source(self, device_name: str, source_name: Optional[str] = None) -> bool:
        """
        Configure the OBS audio input source device.
        """
        target_source = source_name or self.source_name
        logger.info(f"Setting OBS audio source '{target_source}' to audio device '{device_name}'")

        res = self.client.send_request("SetInputSettings", {
            "inputName": target_source,
            "inputSettings": {
                "device_id": device_name
            },
            "overlay": True
        })

        status = res.get("requestStatus", {})
        if status.get("result", False):
            logger.info(f"Successfully updated input settings for '{target_source}'.")
            return True
        else:
            logger.error(f"Failed to set audio input source: {status.get('comment')}")
            return False

    def enable_monitoring(self, source_name: Optional[str] = None, allow_output: bool = True) -> bool:
        """
        Enable audio monitoring for an OBS audio source.
        """
        target_source = source_name or self.source_name
        monitor_type = self.MONITORING_TYPES["MONITOR_AND_OUTPUT"] if allow_output else self.MONITORING_TYPES["MONITOR_ONLY"]

        logger.info(f"Enabling monitoring ({monitor_type}) for source '{target_source}'")
        res = self.client.send_request("SetInputAudioMonitorType", {
            "inputName": target_source,
            "monitorType": monitor_type
        })

        status = res.get("requestStatus", {})
        if status.get("result", False):
            logger.info(f"Enabled monitoring for '{target_source}'.")
            return True
        else:
            logger.error(f"Failed to set monitoring type: {status.get('comment')}")
            return False

    def disable_monitoring(self, source_name: Optional[str] = None) -> bool:
        """
        Disable audio monitoring for an OBS audio source.
        """
        target_source = source_name or self.source_name
        logger.info(f"Disabling monitoring for source '{target_source}'")

        res = self.client.send_request("SetInputAudioMonitorType", {
            "inputName": target_source,
            "monitorType": self.MONITORING_TYPES["NONE"]
        })

        status = res.get("requestStatus", {})
        if status.get("result", False):
            logger.info(f"Disabled audio monitoring for '{target_source}'.")
            return True
        else:
            logger.error(f"Failed to disable monitoring: {status.get('comment')}")
            return False

    def get_audio_monitoring_state(self, source_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve current audio monitoring state for an OBS source.
        """
        target_source = source_name or self.source_name
        res = self.client.send_request("GetInputAudioMonitorType", {
            "inputName": target_source
        })

        status = res.get("requestStatus", {})
        if status.get("result", False):
            mon_type = res.get("responseData", {}).get("monitorType", "UNKNOWN")
            is_enabled = mon_type != self.MONITORING_TYPES["NONE"]
            return {
                "source": target_source,
                "monitor_type": mon_type,
                "monitoring_enabled": is_enabled
            }
        else:
            logger.error(f"Failed to get audio monitoring state: {status.get('comment')}")
            return {"source": target_source, "error": status.get('comment')}

    def set_volume(self, source_name: Optional[str] = None, volume_db: float = 0.0) -> bool:
        """
        Set volume for an OBS audio source in dB.
        """
        target_source = source_name or self.source_name
        logger.info(f"Setting volume for '{target_source}' to {volume_db} dB")

        res = self.client.send_request("SetInputVolume", {
            "inputName": target_source,
            "inputVolumeDb": float(volume_db)
        })

        status = res.get("requestStatus", {})
        if status.get("result", False):
            logger.info(f"Volume for '{target_source}' set to {volume_db} dB.")
            return True
        else:
            logger.error(f"Failed to set volume: {status.get('comment')}")
            return False


# Global default instance
_obs_controller = OBSController()


def connect(host: str = "localhost", port: int = 4455, password: str = "") -> bool:
    """Global connect function wrapper."""
    _obs_controller.host = host
    _obs_controller.port = port
    _obs_controller.password = password
    return _obs_controller.connect()


def set_audio_input_source(device_name: str, source_name: Optional[str] = None) -> bool:
    """Global wrapper for set_audio_input_source."""
    return _obs_controller.set_audio_input_source(device_name, source_name)


def enable_monitoring(source_name: Optional[str] = None) -> bool:
    """Global wrapper for enable_monitoring."""
    return _obs_controller.enable_monitoring(source_name)


def disable_monitoring(source_name: Optional[str] = None) -> bool:
    """Global wrapper for disable_monitoring."""
    return _obs_controller.disable_monitoring(source_name)


def get_audio_monitoring_state(source_name: Optional[str] = None) -> Dict[str, Any]:
    """Global wrapper for get_audio_monitoring_state."""
    return _obs_controller.get_audio_monitoring_state(source_name)


def set_volume(source_name: Optional[str] = None, volume_db: float = 0.0) -> bool:
    """Global wrapper for set_volume."""
    return _obs_controller.set_volume(source_name, volume_db)


if __name__ == "__main__":
    print("Testing OBS Integration...")
    connected = connect()
    if connected:
        print("Monitoring State:", get_audio_monitoring_state("Mic/Aux"))
        _obs_controller.disconnect()
    else:
        print("Could not connect to OBS. Verify OBS is running with WebSocket server enabled.")
