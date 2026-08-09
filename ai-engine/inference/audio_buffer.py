"""
Thread-Safe Circular Audio Buffer for Low-Latency Streaming.

Provides non-blocking and blocking read/write ring buffer operations
over NumPy floating-point audio data.
"""

import logging
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class CircularAudioBuffer:
    """
    Thread-safe circular ring buffer for real-time streaming PCM audio frames.
    Supports single-channel (1D) and multi-channel (2D) audio arrays.
    """

    def __init__(self, capacity: int = 8192, channels: int = 1, dtype: type = np.float32):
        """
        Initialize circular audio buffer.

        Args:
            capacity: Maximum number of audio frames stored in the ring buffer
            channels: Number of audio channels (1 for mono, 2 for stereo)
            dtype: Data type of numpy array elements (default float32)
        """
        self.capacity = capacity
        self.channels = channels
        self.dtype = dtype

        if channels > 1:
            self._buffer = np.zeros((capacity, channels), dtype=dtype)
        else:
            self._buffer = np.zeros(capacity, dtype=dtype)

        self._head = 0  # Write index pointer
        self._tail = 0  # Read index pointer
        self._count = 0  # Current frame count stored

        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    def write(self, data: np.ndarray) -> int:
        """
        Write new audio frames into the circular buffer.
        Overwrites oldest unread samples if write size exceeds available write space.

        Args:
            data: NumPy array of audio frames to write

        Returns:
            Number of frames successfully written
        """
        if data is None or len(data) == 0:
            return 0

        data = np.asarray(data, dtype=self.dtype)
        num_frames = len(data)

        with self._condition:
            if num_frames > self.capacity:
                # Truncate input if single write is larger than entire buffer
                data = data[-self.capacity:]
                num_frames = self.capacity

            # Calculate space available without overwrite
            available_space = self.capacity - self._count
            if num_frames > available_space:
                # Drop oldest unread frames to make space
                overflow = num_frames - available_space
                self._tail = (self._tail + overflow) % self.capacity
                self._count -= overflow

            # First block write up to end of buffer array
            first_chunk = min(num_frames, self.capacity - self._head)
            self._buffer[self._head : self._head + first_chunk] = data[:first_chunk]

            # Second block write wrapped around to beginning
            second_chunk = num_frames - first_chunk
            if second_chunk > 0:
                self._buffer[0:second_chunk] = data[first_chunk:]

            self._head = (self._head + num_frames) % self.capacity
            self._count += num_frames

            self._condition.notify_all()
            return num_frames

    def read(self, num_frames: int, fill_padding: bool = False) -> np.ndarray:
        """
        Read up to `num_frames` audio frames from the buffer.

        Args:
            num_frames: Requested number of frames to read
            fill_padding: If True, pad output with zeros if available frames < num_frames

        Returns:
            NumPy array containing read audio frames
        """
        with self._condition:
            frames_to_read = min(num_frames, self._count)

            if frames_to_read == 0:
                if fill_padding:
                    if self.channels > 1:
                        return np.zeros((num_frames, self.channels), dtype=self.dtype)
                    return np.zeros(num_frames, dtype=self.dtype)
                else:
                    if self.channels > 1:
                        return np.zeros((0, self.channels), dtype=self.dtype)
                    return np.zeros(0, dtype=self.dtype)

            # First block read up to end of buffer array
            first_chunk = min(frames_to_read, self.capacity - self._tail)
            chunk1 = self._buffer[self._tail : self._tail + first_chunk]

            # Second block read wrapped around from beginning
            second_chunk = frames_to_read - first_chunk
            if second_chunk > 0:
                chunk2 = self._buffer[0:second_chunk]
                result = np.concatenate([chunk1, chunk2], axis=0)
            else:
                result = np.copy(chunk1)

            self._tail = (self._tail + frames_to_read) % self.capacity
            self._count -= frames_to_read

            if fill_padding and frames_to_read < num_frames:
                missing = num_frames - frames_to_read
                if self.channels > 1:
                    padding = np.zeros((missing, self.channels), dtype=self.dtype)
                else:
                    padding = np.zeros(missing, dtype=self.dtype)
                result = np.concatenate([result, padding], axis=0)

            return result

    def clear(self) -> None:
        """Reset buffer pointers and wipe frame count."""
        with self._condition:
            self._head = 0
            self._tail = 0
            self._count = 0
            self._buffer.fill(0)
            self._condition.notify_all()

    def available_read(self) -> int:
        """Get total unread frames available for reading."""
        with self._lock:
            return self._count

    def available_write(self) -> int:
        """Get total remaining frame capacity available for writing."""
        with self._lock:
            return self.capacity - self._count

    def is_full(self) -> bool:
        """Check if buffer has reached capacity."""
        with self._lock:
            return self._count >= self.capacity

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        with self._lock:
            return self._count == 0
