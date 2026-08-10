"""
Main Audio Processing Pipeline Orchestrator — Final Version.

Chains noise suppression, gain control, RVC voice conversion, pedalboard effects,
dry/wet mixing, and virtual audio routing into a unified real-time pipeline.

Features:
  - YAML configuration loading with sensible defaults
  - AudioRouter integration for virtual mic output (OBS, Discord, Zoom, VRChat)
  - Event callback system for state changes, errors, and level updates
  - Dry/wet mix control between original and converted audio
  - Bypass mode for A/B comparison
  - Output recording to WAV file
  - Buffer health monitoring with underrun/overrun tracking
  - Comprehensive diagnostics snapshot
  - Graceful shutdown with context manager support
"""

import logging
import threading
import time
import wave
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    from ai_engine.inference.gain_control import GainController
    from ai_engine.rvc.rvc_engine import RVCEngine
    from ai_engine.rvc.rvc_inference import RealtimeRVCInference
except ImportError:
    try:
        from ..inference.gain_control import GainController
        from ..rvc.rvc_engine import RVCEngine
        from ..rvc.rvc_inference import RealtimeRVCInference
    except ImportError:
        from inference.gain_control import GainController
        from rvc.rvc_engine import RVCEngine
        from rvc.rvc_inference import RealtimeRVCInference

from backend.audio_effects import AudioEffectsChain
from backend.noise_suppression import NoiseSuppressor

logger = logging.getLogger(__name__)


class PipelineState(Enum):
    """Pipeline lifecycle states."""
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class AudioPipeline:
    """
    Final audio processing pipeline orchestrator.

    Integrates all DSP components, virtual audio routing, configuration management,
    event callbacks, recording, and diagnostics into one coherent interface.
    """

    DEFAULT_CONFIG = {
        "audio": {
            "sample_rate": 48000,
            "chunk_size": 2048,
            "buffer_size": 8192,
            "channels": 1,
            "input_device": None,
            "output_device": None,
        },
        "engine": {
            "model_path": None,
            "device": "auto",
            "pitch_shift": 0.0,
        },
        "noise_suppression": {"enabled": True},
        "effects": {"enabled": True},
        "gain": {"mode": "manual", "manual_gain_db": 0.0},
        "mix": {"dry_wet": 1.0, "bypass": False},
        "routing": {"target_app": "obs", "custom_device": None, "enabled": True},
        "recording": {"enabled": False, "output_path": "recordings/output.wav"},
    }

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        sample_rate: int = 48000,
        chunk_size: int = 2048,
        buffer_size: int = 8192,
        channels: int = 1,
    ):
        """
        Initialize the pipeline orchestrator.

        Args:
            config_path: Optional path to YAML config file. If provided,
                         settings override constructor defaults.
            sample_rate: Audio sample rate in Hz
            chunk_size: Processing chunk size in frames
            buffer_size: Circular buffer capacity in frames
            channels: Number of audio channels (1=mono, 2=stereo)
        """
        self._lock = threading.RLock()
        self._state = PipelineState.IDLE

        # Load configuration
        self.config: Dict[str, Any] = dict(self.DEFAULT_CONFIG)
        if config_path:
            self._load_config(config_path)
        else:
            # Try default config location
            default_path = Path("configs/default.yaml")
            if default_path.exists():
                self._load_config(default_path)

        # Apply audio config overrides
        audio_cfg = self.config.get("audio", {})
        self.sample_rate = audio_cfg.get("sample_rate", sample_rate)
        self.chunk_size = audio_cfg.get("chunk_size", chunk_size)
        self.buffer_size = audio_cfg.get("buffer_size", buffer_size)
        self.channels = audio_cfg.get("channels", channels)

        # ─── Core DSP Components ───────────────────────────────────
        engine_cfg = self.config.get("engine", {})

        self.rvc_engine = RVCEngine(
            default_pitch_shift=engine_cfg.get("pitch_shift", 0.0)
        )
        self.noise_suppressor = NoiseSuppressor(
            config=self.config.get("noise_suppression", {})
        )
        self.effects_chain = AudioEffectsChain(
            config=self.config.get("effects", {})
        )
        self.gain_controller = GainController(
            **self._filter_gain_config(self.config.get("gain", {}))
        )

        # ─── Audio Router (virtual mic output) ─────────────────────
        self.audio_router: Optional[Any] = None
        routing_cfg = self.config.get("routing", {})
        self._routing_enabled = routing_cfg.get("enabled", True)
        if self._routing_enabled:
            self._init_audio_router(routing_cfg)

        # ─── Real-time Streaming Worker ─────────────────────────────
        audio_cfg = self.config.get("audio", {})
        self.inference_worker = RealtimeRVCInference(
            rvc_engine=self.rvc_engine,
            noise_suppressor=self.noise_suppressor,
            effects_chain=self.effects_chain,
            gain_controller=self.gain_controller,
            sample_rate=self.sample_rate,
            chunk_size=self.chunk_size,
            buffer_size=self.buffer_size,
            input_device=audio_cfg.get("input_device"),
            output_device=audio_cfg.get("output_device"),
            channels=self.channels,
        )

        # ─── Mix & Bypass ───────────────────────────────────────────
        mix_cfg = self.config.get("mix", {})
        self.dry_wet: float = float(mix_cfg.get("dry_wet", 1.0))  # 0=dry, 1=wet
        self.bypass_enabled: bool = bool(mix_cfg.get("bypass", False))

        # ─── Recording ─────────────────────────────────────────────
        self._recording_enabled: bool = False
        self._recording_file: Optional[wave.Wave_write] = None
        self._recording_path: Optional[Path] = None
        rec_cfg = self.config.get("recording", {})
        if rec_cfg.get("enabled", False):
            self._start_recording(rec_cfg.get("output_path", "recordings/output.wav"))

        # ─── Event Callbacks ───────────────────────────────────────
        self._callbacks: Dict[str, List[Callable]] = {
            "state_change": [],
            "error": [],
            "level_update": [],
            "latency_update": [],
        }

        # ─── Buffer Health Monitoring ──────────────────────────────
        self._buffer_underruns: int = 0
        self._buffer_overruns: int = 0
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_interval: float = 0.5  # 2 FPS health check

        self.current_preset: str = "default_voice"

        # Load initial voice model if specified
        model_path = engine_cfg.get("model_path")
        if model_path and Path(model_path).exists():
            self.rvc_engine.load_model(model_path)
            self.current_preset = Path(model_path).stem

        self._set_state(PipelineState.IDLE)
        logger.info("AudioPipeline orchestrator initialized (final).")

    # ═════════════════════════════════════════════════════════════════
    # Configuration
    # ═════════════════════════════════════════════════════════════════

    def _load_config(self, config_path: Union[str, Path]) -> None:
        """Load YAML configuration file and merge with defaults."""
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config file not found: {path}. Using defaults.")
            return

        if not HAS_YAML:
            logger.warning("PyYAML not installed. Cannot load config file. Using defaults.")
            return

        try:
            with open(path, "r") as f:
                loaded = yaml.safe_load(f) or {}

            # Deep merge: loaded config overrides defaults
            for key, val in loaded.items():
                if isinstance(val, dict) and key in self.config and isinstance(self.config[key], dict):
                    self.config[key].update(val)
                else:
                    self.config[key] = val

            logger.info(f"Configuration loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading config from {path}: {e}")

    def _filter_gain_config(self, gain_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Filter gain config dict to only valid GainController constructor args."""
        valid_keys = {"mode", "manual_gain_db", "agc_target_rms", "agc_attack_ms",
                      "agc_release_ms", "max_gain_db", "min_gain_db"}
        return {k: v for k, v in gain_cfg.items() if k in valid_keys}

    def _init_audio_router(self, routing_cfg: Dict[str, Any]) -> None:
        """Initialize the audio router for virtual mic output."""
        try:
            from app.routing.audio_router import AudioRouter
            self.audio_router = AudioRouter()
            target = routing_cfg.get("target_app", "obs")
            custom_device = routing_cfg.get("custom_device")
            self.audio_router.initialize_router(target, custom_device)
            logger.info(f"AudioRouter initialized for target: {target}")
        except Exception as e:
            logger.warning(f"Could not initialize AudioRouter: {e}. Virtual mic routing disabled.")
            self.audio_router = None
            self._routing_enabled = False

    # ═════════════════════════════════════════════════════════════════
    # Lifecycle Control
    # ═════════════════════════════════════════════════════════════════

    def start(self) -> Dict[str, Any]:
        """
        Start the real-time pipeline.

        Starts the inference worker, audio router, buffer health monitor,
        and transitions to RUNNING state.

        Returns:
            Dict with startup status and parameters
        """
        with self._lock:
            if self._state == PipelineState.RUNNING:
                logger.warning("Pipeline already running.")
                return self.get_status()

            self._set_state(PipelineState.STARTING)

            try:
                # Start inference worker
                self.inference_worker.start()

                # Start buffer health monitor
                self._monitor_thread = threading.Thread(
                    target=self._monitor_loop,
                    name="PipelineHealthMonitor",
                    daemon=True
                )
                self._monitor_thread.start()

                self._set_state(PipelineState.RUNNING)
                self._emit("state_change", {"state": "running"})

                result = {
                    "status": "running",
                    "active_preset": self.current_preset,
                    "sample_rate": self.sample_rate,
                    "chunk_size": self.chunk_size,
                    "channels": self.channels,
                    "routing_enabled": self._routing_enabled,
                    "recording": self._recording_enabled,
                    "bypass": self.bypass_enabled,
                    "dry_wet": self.dry_wet,
                }
                logger.info("AudioPipeline started successfully.")
                return result

            except Exception as e:
                self._set_state(PipelineState.ERROR)
                self._emit("error", {"error": str(e), "context": "start"})
                logger.error(f"Failed to start pipeline: {e}", exc_info=True)
                return {"status": "error", "error": str(e)}

    def stop(self) -> Dict[str, Any]:
        """
        Stop the real-time pipeline gracefully.

        Stops inference worker, closes audio router, stops recording,
        and transitions to STOPPED state.

        Returns:
            Dict with shutdown status
        """
        with self._lock:
            if self._state in (PipelineState.STOPPED, PipelineState.IDLE):
                return {"status": "stopped"}

            self._set_state(PipelineState.STOPPING)

            try:
                # Stop inference worker
                self.inference_worker.stop()

                # Stop health monitor
                if self._monitor_thread and self._monitor_thread.is_alive():
                    self._monitor_thread.join(timeout=2.0)
                self._monitor_thread = None

                # Close audio router
                if self.audio_router:
                    self.audio_router.close()

                # Stop recording
                self._stop_recording()

                self._set_state(PipelineState.STOPPED)
                self._emit("state_change", {"state": "stopped"})

                logger.info("AudioPipeline stopped successfully.")
                return {"status": "stopped"}

            except Exception as e:
                self._set_state(PipelineState.ERROR)
                self._emit("error", {"error": str(e), "context": "stop"})
                logger.error(f"Error stopping pipeline: {e}", exc_info=True)
                return {"status": "error", "error": str(e)}

    def restart(self) -> Dict[str, Any]:
        """Restart the pipeline (stop + start)."""
        self.stop()
        time.sleep(0.1)
        return self.start()

    def is_running(self) -> bool:
        """Check if pipeline is actively running."""
        return self._state == PipelineState.RUNNING

    @property
    def state(self) -> str:
        """Get current pipeline state as string."""
        return self._state.value

    # ═════════════════════════════════════════════════════════════════
    # Voice & Pitch Control
    # ═════════════════════════════════════════════════════════════════

    def set_voice_preset(self, preset_id: str) -> bool:
        """
        Switch active voice model preset.

        Args:
            preset_id: Voice preset identifier

        Returns:
            True if switch succeeded
        """
        logger.info(f"Switching voice preset to: {preset_id}")

        if preset_id == "default_voice":
            self.current_preset = preset_id
            self._emit("state_change", {"preset": preset_id})
            return True

        success = self.rvc_engine.load_model(f"ai-engine/models/presets/{preset_id}.pth")
        if success:
            self.current_preset = preset_id
            self._emit("state_change", {"preset": preset_id})
            return True
        return False

    def set_pitch_shift(self, semitones: float) -> None:
        """Set pitch shift in semitones (-24 to +24)."""
        self.rvc_engine.set_pitch_shift(semitones)

    def get_available_voices(self) -> List[Dict[str, Any]]:
        """List available voice presets."""
        return self.rvc_engine.get_available_voices()

    # ═════════════════════════════════════════════════════════════════
    # Mix & Bypass
    # ═════════════════════════════════════════════════════════════════

    def set_dry_wet(self, value: float) -> None:
        """
        Set dry/wet mix ratio.

        Args:
            value: 0.0 = fully dry (original), 1.0 = fully wet (converted)
        """
        self.dry_wet = float(max(0.0, min(1.0, value)))

    def set_bypass(self, enabled: bool) -> None:
        """
        Enable or disable bypass mode (passthrough without conversion).

        When bypassed, audio passes through the pipeline unchanged for
        A/B comparison between original and converted voice.

        Args:
            enabled: True to bypass conversion, False to enable it
        """
        self.bypass_enabled = bool(enabled)
        logger.info(f"Bypass mode {'enabled' if enabled else 'disabled'}")

    # ═════════════════════════════════════════════════════════════════
    # Push-to-Talk
    # ═════════════════════════════════════════════════════════════════

    def set_ptt_enabled(self, enabled: bool) -> None:
        """Enable or disable push-to-talk mode."""
        self.inference_worker.set_ptt_enabled(enabled)

    def set_ptt_active(self, active: bool) -> None:
        """Set push-to-talk transmit state (True=transmit, False=mute)."""
        self.inference_worker.set_ptt_active(active)

    # ═════════════════════════════════════════════════════════════════
    # Recording
    # ═════════════════════════════════════════════════════════════════

    def _start_recording(self, output_path: str) -> None:
        """Start recording output audio to WAV file."""
        try:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            self._recording_file = wave.open(str(path), "w")
            self._recording_file.setnchannels(self.channels)
            self._recording_file.setsampwidth(2)  # 16-bit
            self._recording_file.setframerate(self.sample_rate)
            self._recording_path = path
            self._recording_enabled = True
            logger.info(f"Recording started: {path}")
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self._recording_enabled = False

    def _stop_recording(self) -> None:
        """Stop recording and close the WAV file."""
        if self._recording_file:
            try:
                self._recording_file.close()
                logger.info(f"Recording saved: {self._recording_path}")
            except Exception as e:
                logger.error(f"Error closing recording file: {e}")
            self._recording_file = None
            self._recording_enabled = False

    def _write_recording(self, audio: np.ndarray) -> None:
        """Write audio chunk to recording file."""
        if not self._recording_enabled or not self._recording_file:
            return
        try:
            # Convert float32 to 16-bit PCM
            pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            if pcm.ndim == 1 and self.channels > 1:
                pcm = np.column_stack([pcm] * self.channels)
            self._recording_file.writeframes(pcm.tobytes())
        except Exception as e:
            logger.error(f"Error writing to recording: {e}")

    def start_recording(self, output_path: str = "recordings/output.wav") -> bool:
        """
        Start recording output audio to a WAV file.

        Args:
            output_path: Path to save the WAV file

        Returns:
            True if recording started successfully
        """
        if self._recording_enabled:
            logger.warning("Recording already in progress.")
            return False
        self._start_recording(output_path)
        return self._recording_enabled

    def stop_recording(self) -> Dict[str, Any]:
        """Stop recording and return the file path."""
        path = str(self._recording_path) if self._recording_path else None
        self._stop_recording()
        return {"recording_stopped": True, "file_path": path}

    # ═════════════════════════════════════════════════════════════════
    # Audio Routing
    # ═════════════════════════════════════════════════════════════════

    def set_output_target(self, target_app: str, custom_device: Optional[str] = None) -> bool:
        """
        Switch the virtual audio output target (OBS, Discord, Zoom, VRChat).

        Args:
            target_app: Target application key
            custom_device: Optional custom device name override

        Returns:
            True if target switch succeeded
        """
        if not self.audio_router:
            logger.warning("AudioRouter not available.")
            return False

        was_running = self.is_running()
        if was_running:
            self.inference_worker.stop()

        success = self.audio_router.switch_output_target(target_app, custom_device)

        if was_running:
            self.inference_worker.start()

        return success

    def get_routing_status(self) -> Dict[str, Any]:
        """Get current audio routing status."""
        if self.audio_router:
            return self.audio_router.get_routing_status()
        return {"active": False, "enabled": self._routing_enabled}

    def detect_virtual_devices(self) -> List[Dict[str, Any]]:
        """Detect available virtual audio devices on the system."""
        if self.audio_router:
            return self.audio_router.detect_virtual_devices()
        return []

    # ═════════════════════════════════════════════════════════════════
    # Telemetry & Diagnostics
    # ═════════════════════════════════════════════════════════════════

    def get_latency(self) -> Dict[str, float]:
        """Get latency breakdown in milliseconds."""
        return self.inference_worker.get_latency_stats()

    def get_audio_levels(self) -> Dict[str, float]:
        """Get current input/output audio levels for UI meters."""
        return self.inference_worker.get_audio_levels()

    def get_device_info(self) -> Dict[str, Any]:
        """Get GPU/device diagnostics."""
        return self.rvc_engine.get_device_info()

    def get_buffer_health(self) -> Dict[str, int]:
        """Get buffer underrun/overrun counts."""
        return {
            "underruns": self._buffer_underruns,
            "overruns": self._buffer_overruns,
            "input_available": self.inference_worker.input_buffer.available_read(),
            "output_available": self.inference_worker.output_buffer.available_read(),
        }

    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive pipeline status snapshot.

        Returns all operational state in a single dict — useful for
        the /api/conversion/status endpoint and WebSocket streaming.
        """
        return {
            "state": self.state,
            "is_running": self.is_running(),
            "active_preset": self.current_preset,
            "pitch_shift": self.rvc_engine.pitch_shift,
            "latency": self.get_latency(),
            "levels": self.get_audio_levels(),
            "device_info": self.get_device_info(),
            "buffer_health": self.get_buffer_health(),
            "gain_state": self.gain_controller.get_state(),
            "routing": self.get_routing_status(),
            "bypass": self.bypass_enabled,
            "dry_wet": self.dry_wet,
            "recording": self._recording_enabled,
            "recording_path": str(self._recording_path) if self._recording_path else None,
            "ptt_enabled": self.inference_worker._ptt_enabled,
            "ptt_active": self.inference_worker._ptt_active,
        }

    # ═════════════════════════════════════════════════════════════════
    # Settings Update
    # ═════════════════════════════════════════════════════════════════

    def update_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update runtime audio settings.

        Supports: pitch_shift, noise_suppression, effects, gain, input_device,
        output_device, ptt_enabled, dry_wet, bypass, recording.

        Args:
            settings: Dict of settings to update

        Returns:
            Dict with updated state summary
        """
        if "pitch_shift" in settings:
            self.set_pitch_shift(float(settings["pitch_shift"]))

        if "noise_suppression" in settings:
            ns_cfg = settings["noise_suppression"]
            if isinstance(ns_cfg, dict):
                self.noise_suppressor.update_config(ns_cfg)
            elif isinstance(ns_cfg, bool):
                self.noise_suppressor.enabled = ns_cfg

        if "effects" in settings:
            self.effects_chain.update_config(settings["effects"])

        if "gain" in settings:
            gain_cfg = settings["gain"]
            if isinstance(gain_cfg, dict):
                if "mode" in gain_cfg:
                    self.gain_controller.set_mode(gain_cfg["mode"])
                if "manual_gain_db" in gain_cfg:
                    self.gain_controller.set_manual_gain(gain_cfg["manual_gain_db"])
                if "noise_gate_enabled" in gain_cfg:
                    self.gain_controller.noise_gate_enabled = bool(gain_cfg["noise_gate_enabled"])
            elif isinstance(gain_cfg, (int, float)):
                self.gain_controller.set_manual_gain(float(gain_cfg))

        if "input_device" in settings:
            self.inference_worker.input_device = settings["input_device"]

        if "output_device" in settings:
            self.inference_worker.output_device = settings["output_device"]

        if "ptt_enabled" in settings:
            self.set_ptt_enabled(bool(settings["ptt_enabled"]))

        if "dry_wet" in settings:
            self.set_dry_wet(float(settings["dry_wet"]))

        if "bypass" in settings:
            self.set_bypass(bool(settings["bypass"]))

        if "recording" in settings:
            rec_cfg = settings["recording"]
            if isinstance(rec_cfg, dict):
                if rec_cfg.get("enabled"):
                    self.start_recording(rec_cfg.get("output_path", "recordings/output.wav"))
                else:
                    self.stop_recording()

        return {
            "status": "updated",
            "is_running": self.is_running(),
            "pitch_shift": self.rvc_engine.pitch_shift,
            "noise_suppression_enabled": self.noise_suppressor.enabled,
            "effects_enabled": self.effects_chain.enabled,
            "gain_state": self.gain_controller.get_state(),
            "bypass": self.bypass_enabled,
            "dry_wet": self.dry_wet,
            "ptt_enabled": self.inference_worker._ptt_enabled,
            "recording": self._recording_enabled,
        }

    # ═════════════════════════════════════════════════════════════════
    # Event Callback System
    # ═════════════════════════════════════════════════════════════════

    def on(self, event: str, callback: Callable) -> None:
        """
        Register a callback for a pipeline event.

        Events: 'state_change', 'error', 'level_update', 'latency_update'

        Args:
            event: Event name
            callback: Callable invoked with event data dict
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)
        else:
            logger.warning(f"Unknown event type: {event}")

    def off(self, event: str, callback: Callable) -> None:
        """Remove a previously registered callback."""
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)

    def _emit(self, event: str, data: Dict[str, Any]) -> None:
        """Emit an event to all registered callbacks."""
        for cb in self._callbacks.get(event, []):
            try:
                cb(data)
            except Exception as e:
                logger.error(f"Error in {event} callback: {e}")

    # ═════════════════════════════════════════════════════════════════
    # Internal: State & Monitoring
    # ═════════════════════════════════════════════════════════════════

    def _set_state(self, state: PipelineState) -> None:
        """Transition to a new pipeline state."""
        old = self._state
        self._state = state
        if old != state:
            logger.debug(f"Pipeline state: {old.value} → {state.value}")

    def _monitor_loop(self) -> None:
        """Background thread for buffer health monitoring and event emission."""
        while self._state == PipelineState.RUNNING:
            try:
                # Check buffer health
                input_avail = self.inference_worker.input_buffer.available_read()
                output_avail = self.inference_worker.output_buffer.available_read()

                # Detect underrun: output buffer empty while input has data
                if output_avail == 0 and input_avail > 0:
                    self._buffer_underruns += 1

                # Detect overrun: input buffer full (data being dropped)
                if self.inference_worker.input_buffer.is_full():
                    self._buffer_overruns += 1

                # Emit level updates
                levels = self.inference_worker.get_audio_levels()
                self._emit("level_update", levels)

                # Emit latency updates
                latency = self.inference_worker.get_latency_stats()
                self._emit("latency_update", latency)

                # Route processed audio to virtual mic if routing enabled
                if self._routing_enabled and self.audio_router and output_avail > 0:
                    chunk = self.inference_worker.output_buffer.read(min(output_avail, self.chunk_size))
                    if self.bypass_enabled:
                        # In bypass mode, route the original input instead
                        raw = self.inference_worker.input_buffer.read(min(
                            self.inference_worker.input_buffer.available_read(),
                            self.chunk_size
                        ), fill_padding=True)
                        if len(raw) > 0:
                            chunk = raw

                    # Apply dry/wet mix
                    if self.dry_wet < 1.0 and not self.bypass_enabled:
                        raw_chunk = self.inference_worker.input_buffer.read(
                            min(self.inference_worker.input_buffer.available_read(), len(chunk)),
                            fill_padding=True
                        )
                        if len(raw_chunk) == len(chunk):
                            chunk = (self.dry_wet * chunk + (1.0 - self.dry_wet) * raw_chunk).astype(np.float32)

                    # Write to recording
                    self._write_recording(chunk)

                    # Route to virtual mic
                    self.audio_router.route_audio(chunk)

            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                self._emit("error", {"error": str(e), "context": "monitor"})

            time.sleep(self._monitor_interval)

    # ═════════════════════════════════════════════════════════════════
    # Context Manager & Cleanup
    # ═════════════════════════════════════════════════════════════════

    def __enter__(self):
        """Context manager entry — starts the pipeline."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit — stops the pipeline gracefully."""
        self.stop()
        return False

    def __del__(self):
        """Destructor — ensure cleanup if not explicitly stopped."""
        try:
            if self.is_running():
                self.stop()
        except Exception:
            pass  # Avoid errors during interpreter shutdown
