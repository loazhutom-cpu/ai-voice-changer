"""
Real-time RVC Inference Loop and Stream Handler.

Captures audio from an input device in low-latency chunks, passes through
noise suppression, gain control, RVC voice conversion, and audio effects,
then writes output to a virtual audio device.
"""

import logging
import threading
import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

try:
    import sounddevice as sd
    HAS_SD = True
except ImportError:
    HAS_SD = False

try:
    from ai_engine.inference.audio_buffer import CircularAudioBuffer
    from ai_engine.inference.gain_control import GainController
    from ai_engine.rvc.rvc_engine import RVCEngine
except ImportError:
    try:
        from ..inference.audio_buffer import CircularAudioBuffer
        from ..inference.gain_control import GainController
        from .rvc_engine import RVCEngine
    except ImportError:
        from inference.audio_buffer import CircularAudioBuffer
        from inference.gain_control import GainController
        from rvc_engine import RVCEngine

logger = logging.getLogger(__name__)


class RealtimeRVCInference:
    """
    Real-time streaming audio conversion worker.

    Uses sounddevice streams, circular buffers, and multi-threaded DSP
    chaining for sub-100ms latency voice conversion.
    """

    def __init__(
        self,
        rvc_engine: RVCEngine,
        noise_suppressor: Optional[Any] = None,
        effects_chain: Optional[Any] = None,
        gain_controller: Optional[GainController] = None,
        sample_rate: int = 48000,
        chunk_size: int = 2048,
        buffer_size: int = 8192,
        input_device: Optional[Union[int, str]] = None,
        output_device: Optional[Union[int, str]] = None,
        channels: int = 1
    ):
        self.rvc_engine = rvc_engine
        self.noise_suppressor = noise_suppressor
        self.effects_chain = effects_chain
        self.gain_controller = gain_controller or GainController()
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.input_device = input_device
        self.output_device = output_device

        # Buffers
        self.input_buffer = CircularAudioBuffer(capacity=buffer_size, channels=channels)
        self.output_buffer = CircularAudioBuffer(capacity=buffer_size, channels=channels)

        # State
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._input_stream: Optional[Any] = None
        self._output_stream: Optional[Any] = None

        # Push-to-talk
        self._ptt_active = False
        self._ptt_enabled = False

        # Level meters
        self.input_level_rms = 0.0
        self.output_level_rms = 0.0

        # Latency tracking (exponential moving average)
        self.processing_latency_ms = 0.0
        self.buffer_latency_ms = 0.0
        self.total_latency_ms = 0.0

        # Stats
        self._frames_processed = 0
        self._frames_dropped = 0
        self._last_process_time = 0.0

    def _input_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """Audio capture callback (runs in audio thread)."""
        if status:
            logger.warning(f"Input stream status: {status}")

        audio_data = indata.copy().astype(np.float32)

        # RMS level for metering
        rms = float(np.sqrt(np.mean(audio_data ** 2) + 1e-9))
        self.input_level_rms = 0.8 * self.input_level_rms + 0.2 * rms

        # Push-to-talk gate: if PTT enabled and not active, discard audio
        if self._ptt_enabled and not self._ptt_active:
            return

        self.input_buffer.write(audio_data)

    def _output_callback(self, outdata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """Audio output callback (runs in audio thread)."""
        if status:
            logger.warning(f"Output stream status: {status}")

        read_data = self.output_buffer.read(frames, fill_padding=True)

        if read_data.ndim == 1 and self.channels > 1:
            read_data = np.column_stack([read_data] * self.channels)

        outdata[:] = read_data.reshape(outdata.shape)

        # Output RMS level
        rms = float(np.sqrt(np.mean(outdata ** 2) + 1e-9))
        self.output_level_rms = 0.8 * self.output_level_rms + 0.2 * rms

    def _processing_loop(self) -> None:
        """Worker thread: noise reduction → gain → RVC → effects → output."""
        logger.info("Real-time audio processing loop started.")

        while self._running:
            if self.input_buffer.available_read() >= self.chunk_size:
                t0 = time.perf_counter()

                # Step 1: Read raw chunk
                chunk = self.input_buffer.read(self.chunk_size)

                # Flatten to mono for processing
                if chunk.ndim > 1:
                    chunk_mono = np.mean(chunk, axis=1)
                else:
                    chunk_mono = chunk

                # Step 2: Noise suppression
                if self.noise_suppressor is not None and getattr(self.noise_suppressor, "enabled", True):
                    try:
                        chunk_mono = self.noise_suppressor.process(chunk_mono, self.sample_rate)
                    except Exception as e:
                        logger.error(f"Noise suppression error: {e}")

                # Step 3: Gain control (input staging)
                try:
                    chunk_mono = self.gain_controller.process(chunk_mono, self.sample_rate)
                except Exception as e:
                    logger.error(f"Gain control error: {e}")

                # Step 4: RVC voice conversion
                try:
                    converted = self.rvc_engine.convert_audio(
                        chunk_mono,
                        sample_rate=self.sample_rate
                    )
                except Exception as e:
                    logger.error(f"RVC conversion error: {e}")
                    converted = chunk_mono

                # Step 5: Audio effects chain
                if self.effects_chain is not None and getattr(self.effects_chain, "enabled", True):
                    try:
                        processed = self.effects_chain.process(converted, self.sample_rate)
                    except Exception as e:
                        logger.error(f"Effects processing error: {e}")
                        processed = converted
                else:
                    processed = converted

                # Step 6: Write to output buffer
                if self.channels > 1 and processed.ndim == 1:
                    processed = np.column_stack([processed] * self.channels)

                self.output_buffer.write(processed)

                # Latency calculation (EMA smoothing)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                self.processing_latency_ms = 0.9 * self.processing_latency_ms + 0.1 * elapsed_ms
                self.buffer_latency_ms = (self.chunk_size / self.sample_rate) * 1000.0
                self.total_latency_ms = self.processing_latency_ms + self.buffer_latency_ms
                self._frames_processed += 1

            else:
                # Not enough data yet — brief sleep to avoid busy-waiting
                time.sleep(0.002)

        logger.info("Real-time audio processing loop stopped.")

    def start(self) -> None:
        """Start audio streams and processing thread."""
        if self._running:
            logger.warning("Inference loop already running.")
            return

        self._running = True
        self._frames_processed = 0
        self._frames_dropped = 0
        self.input_buffer.clear()
        self.output_buffer.clear()

        # Start processing thread
        self._thread = threading.Thread(
            target=self._processing_loop,
            name="RVCInferenceWorker",
            daemon=True
        )
        self._thread.start()

        # Start audio streams
        if HAS_SD:
            try:
                self._input_stream = sd.InputStream(
                    device=self.input_device,
                    channels=self.channels,
                    samplerate=self.sample_rate,
                    blocksize=self.chunk_size,
                    callback=self._input_callback
                )
                self._output_stream = sd.OutputStream(
                    device=self.output_device,
                    channels=self.channels,
                    samplerate=self.sample_rate,
                    blocksize=self.chunk_size,
                    callback=self._output_callback
                )
                self._input_stream.start()
                self._output_stream.start()
                logger.info("Audio streams started successfully.")
            except Exception as e:
                logger.warning(f"Could not open audio streams: {e}. Running in headless mode.")
        else:
            logger.info("sounddevice not available. Running in headless/simulated mode.")

    def stop(self) -> None:
        """Stop audio streams and processing thread."""
        if not self._running:
            return

        self._running = False

        # Stop streams
        if self._input_stream:
            try:
                self._input_stream.stop()
                self._input_stream.close()
            except Exception as e:
                logger.error(f"Error closing input stream: {e}")
            self._input_stream = None

        if self._output_stream:
            try:
                self._output_stream.stop()
                self._output_stream.close()
            except Exception as e:
                logger.error(f"Error closing output stream: {e}")
            self._output_stream = None

        # Join thread
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

        logger.info("Inference engine stopped.")

    def is_running(self) -> bool:
        """Check if pipeline is active."""
        return self._running

    def set_ptt_enabled(self, enabled: bool) -> None:
        """Enable or disable push-to-talk mode."""
        self._ptt_enabled = enabled
        logger.info(f"Push-to-talk {'enabled' if enabled else 'disabled'}")

    def set_ptt_active(self, active: bool) -> None:
        """Set push-to-talk button state (True = transmitting)."""
        self._ptt_active = active

    def get_latency_stats(self) -> Dict[str, float]:
        """
        Get latency breakdown in milliseconds.

        Returns:
            Dict with processing_ms, buffer_ms, total_ms, frames_processed, frames_dropped
        """
        return {
            "processing_ms": round(self.processing_latency_ms, 2),
            "buffer_ms": round(self.buffer_latency_ms, 2),
            "total_ms": round(self.total_latency_ms, 2),
            "frames_processed": self._frames_processed,
            "frames_dropped": self._frames_dropped
        }

    def get_audio_levels(self) -> Dict[str, float]:
        """
        Get current audio levels for UI meters.

        Returns:
            Dict with input/output RMS and dB values
        """
        input_db = 20 * np.log10(self.input_level_rms + 1e-9)
        output_db = 20 * np.log10(self.output_level_rms + 1e-9)
        return {
            "input_rms": round(self.input_level_rms, 6),
            "output_rms": round(self.output_level_rms, 6),
            "input_db": round(float(input_db), 2),
            "output_db": round(float(output_db), 2),
            "is_running": self._running,
            "ptt_enabled": self._ptt_enabled,
            "ptt_active": self._ptt_active
        }
