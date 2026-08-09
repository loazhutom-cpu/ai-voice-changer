"""
Real-time RVC Inference Loop and Stream Handler.

Captures audio from an input device in low-latency chunks, passes it through noise suppression,
RVC voice conversion, and pedalboard audio effects, and writes the output to a virtual device.
"""

import logging
import queue
import threading
import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import sounddevice as sd

try:
    from ai_engine.inference.audio_buffer import CircularAudioBuffer
    from ai_engine.rvc.rvc_engine import RVCEngine
except ImportError:
    try:
        from ..inference.audio_buffer import CircularAudioBuffer
        from .rvc_engine import RVCEngine
    except ImportError:
        from inference.audio_buffer import CircularAudioBuffer
        from rvc_engine import RVCEngine

logger = logging.getLogger(__name__)


class RealtimeRVCInference:
    """
    Real-time streaming audio conversion worker using sounddevice streams,
    circular buffers, and low-latency DSP chaining.
    """

    def __init__(
        self,
        rvc_engine: RVCEngine,
        noise_suppressor: Optional[Any] = None,
        effects_chain: Optional[Any] = None,
        sample_rate: int = 48000,
        chunk_size: int = 2048,  # ~42.6 ms at 48kHz
        buffer_size: int = 8192,
        input_device: Optional[Union[int, str]] = None,
        output_device: Optional[Union[int, str]] = None,
        channels: int = 1
    ):
        """
        Initialize Realtime Inference worker.

        Args:
            rvc_engine: Active RVCEngine instance
            noise_suppressor: Optional NoiseSuppressor instance
            effects_chain: Optional AudioEffectsChain instance
            sample_rate: Sampling frequency in Hz (default: 48000)
            chunk_size: Frames per callback chunk (~40ms latency window)
            buffer_size: Max capacity of circular buffer
            input_device: Sounddevice index or name for mic input
            output_device: Sounddevice index or name for virtual mic output
            channels: Number of audio channels (1 for mono, 2 for stereo)
        """
        self.rvc_engine = rvc_engine
        self.noise_suppressor = noise_suppressor
        self.effects_chain = effects_chain
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels

        self.input_device = input_device
        self.output_device = output_device

        # Low-latency thread-safe audio buffers
        self.input_buffer = CircularAudioBuffer(capacity=buffer_size, channels=channels)
        self.output_buffer = CircularAudioBuffer(capacity=buffer_size, channels=channels)

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._input_stream: Optional[sd.InputStream] = None
        self._output_stream: Optional[sd.OutputStream] = None

        # Level meters and performance tracking
        self.input_level_rms: float = 0.0
        self.output_level_rms: float = 0.0
        self.processing_latency_ms: float = 0.0
        self.last_process_time: float = 0.0

    def _input_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags) -> None:
        """Sounddevice audio capture callback (runs in high-priority audio thread)."""
        if status:
            logger.warning(f"Input stream warning: {status}")

        audio_data = indata.copy().astype(np.float32)

        # Calculate RMS level for metering
        rms = np.sqrt(np.mean(audio_data ** 2) + 1e-9)
        self.input_level_rms = float(rms)

        # Write to input circular buffer
        self.input_buffer.write(audio_data)

    def _output_callback(self, outdata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags) -> None:
        """Sounddevice audio output callback (runs in high-priority audio thread)."""
        if status:
            logger.warning(f"Output stream warning: {status}")

        read_data = self.output_buffer.read(frames)
        if len(read_data) < frames:
            # Underflow recovery: pad remaining frames with silence
            pad_length = frames - len(read_data)
            padding = np.zeros((pad_length, self.channels), dtype=np.float32) if self.channels > 1 else np.zeros(pad_length, dtype=np.float32)
            if len(read_data) > 0:
                read_data = np.concatenate([read_data, padding])
            else:
                read_data = padding

        if read_data.ndim == 1 and self.channels > 1:
            read_data = np.column_stack([read_data] * self.channels)

        outdata[:] = read_data.reshape(outdata.shape)

        # Calculate RMS output level
        rms = np.sqrt(np.mean(outdata ** 2) + 1e-9)
        self.output_level_rms = float(rms)

    def _processing_loop(self) -> None:
        """Worker thread processing loop: noise reduction -> RVC -> effects."""
        logger.info("Real-time audio processing loop started.")

        while self._running:
            if self.input_buffer.available_read() >= self.chunk_size:
                t0 = time.perf_counter()

                # Step 1: Read raw chunk from input buffer
                chunk = self.input_buffer.read(self.chunk_size)

                # Ensure mono 1D format for DSP processing
                if chunk.ndim > 1:
                    chunk_mono = np.mean(chunk, axis=1)
                else:
                    chunk_mono = chunk

                # Step 2: Noise Suppression
                if self.noise_suppressor is not None and getattr(self.noise_suppressor, "enabled", True):
                    try:
                        chunk_mono = self.noise_suppressor.process(chunk_mono, self.sample_rate)
                    except Exception as e:
                        logger.error(f"Error in noise suppression: {e}")

                # Step 3: RVC Voice Conversion
                try:
                    converted = self.rvc_engine.convert_audio(
                        chunk_mono,
                        sample_rate=self.sample_rate
                    )
                except Exception as e:
                    logger.error(f"Error in RVC conversion: {e}")
                    converted = chunk_mono

                # Step 4: Audio Effects Chain (Compressor, Reverb, EQ, Limiter)
                if self.effects_chain is not None and getattr(self.effects_chain, "enabled", True):
                    try:
                        processed = self.effects_chain.process(converted, self.sample_rate)
                    except Exception as e:
                        logger.error(f"Error in effects processing: {e}")
                        processed = converted
                else:
                    processed = converted

                # Step 5: Write processed audio to output buffer
                if self.channels > 1 and processed.ndim == 1:
                    processed = np.column_stack([processed] * self.channels)

                self.output_buffer.write(processed)

                # Performance metric calculation
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                self.processing_latency_ms = 0.9 * self.processing_latency_ms + 0.1 * elapsed_ms
            else:
                time.sleep(0.005)

        logger.info("Real-time audio processing loop stopped.")

    def start(self) -> None:
        """Start sounddevice streams and background processing thread."""
        if self._running:
            logger.warning("Inference loop is already running.")
            return

        self._running = True
        self.input_buffer.clear()
        self.output_buffer.clear()

        # Initialize background processing thread
        self._thread = threading.Thread(target=self._processing_loop, name="RVCInferenceWorker", daemon=True)
        self._thread.start()

        # Initialize Sounddevice Audio Streams
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
            logger.info("Audio hardware streams started successfully.")
        except Exception as e:
            logger.warning(f"Could not open sounddevice audio hardware streams: {e}. Running in headless/simulated audio mode.")

    def stop(self) -> None:
        """Stop sounddevice streams and worker thread."""
        if not self._running:
            return

        self._running = False

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

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            self._thread = None

        logger.info("Inference engine stopped successfully.")

    def is_running(self) -> bool:
        """Return active status of the pipeline."""
        return self._running

    def get_latency_stats(self) -> Dict[str, float]:
        """
        Calculate total pipeline latency components in milliseconds.

        Returns:
            Dict containing buffer_ms, processing_ms, total_latency_ms
        """
        buffer_ms = (self.chunk_size / self.sample_rate) * 1000.0
        return {
            "chunk_buffer_ms": round(buffer_ms, 2),
            "processing_ms": round(self.processing_latency_ms, 2),
            "total_latency_ms": round(buffer_ms * 2 + self.processing_latency_ms, 2)
        }

    def get_audio_levels(self) -> Dict[str, float]:
        """
        Get current input and output RMS levels (scaled 0.0 to 1.0).

        Returns:
            Dict containing input_level and output_level
        """
        return {
            "input_level": round(float(np.clip(self.input_level_rms * 5.0, 0.0, 1.0)), 4),
            "output_level": round(float(np.clip(self.output_level_rms * 5.0, 0.0, 1.0)), 4),
            "input_db": round(float(20 * np.log10(self.input_level_rms + 1e-6)), 2),
            "output_db": round(float(20 * np.log10(self.output_level_rms + 1e-6)), 2)
        }
