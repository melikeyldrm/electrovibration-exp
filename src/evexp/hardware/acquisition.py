"""Continuous acquisition on a worker thread.

10 kHz device reads run on their own thread so the Qt event loop never
blocks; the worker writes a ring buffer and a snapshot, and the Qt timer
(~30 Hz) only reads that snapshot under a lock - no Qt objects cross it.

Position is also sampled here, once per block, so it shares a clock with
the force channels and can later be cut from the same time window.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np

from evexp.hardware.base import DAQDevice, SensorChunk, default_chunk_samples
from evexp.hardware.position import PositionSample, PositionSource


@dataclass(frozen=True)
class AcquisitionStats:
    """Read-loop health. lag_s growing steadily means the loop is falling
    behind the device - worth catching before the driver buffer overflows.
    """

    running: bool
    chunks_read: int
    samples_read: int
    slow_reads: int
    mean_read_s: float
    max_read_s: float
    elapsed_s: float
    lag_s: float
    last_error: Optional[str]

    def describe(self) -> str:
        state = "running" if self.running else "stopped"
        return (
            f"acquisition {state}: {self.samples_read} samples in "
            f"{self.chunks_read} chunks over {self.elapsed_s:.1f} s, "
            f"read {self.mean_read_s * 1000:.1f} ms mean / "
            f"{self.max_read_s * 1000:.1f} ms worst, "
            f"{self.slow_reads} slow, lag {self.lag_s * 1000:+.0f} ms"
        )


class RingBuffer:
    """Fixed-capacity circular store for multi-channel samples.

    A window onto the recent past, not the session record - old data is
    overwritten. Absolute sample indices are tracked so a stretch can be
    addressed by when it happened, not where it landed.
    """

    def __init__(self, n_channels: int, capacity_samples: int):
        if n_channels <= 0 or capacity_samples <= 0:
            raise ValueError("ring buffer needs a positive size")
        self._data = np.zeros((n_channels, capacity_samples))
        self._capacity = capacity_samples
        self._write_at = 0
        self._total_written = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def total_written(self) -> int:
        """Samples ever written, including those since overwritten."""
        return self._total_written

    @property
    def oldest_index(self) -> int:
        """Absolute index of the oldest sample still held."""
        return max(0, self._total_written - self._capacity)

    def write(self, block: np.ndarray) -> None:
        if block.shape[0] != self._data.shape[0]:
            raise ValueError(
                f"expected {self._data.shape[0]} channels, got {block.shape[0]}")

        n = block.shape[1]
        if n >= self._capacity:
            # A block larger than the whole buffer: keep only its tail.
            self._data[:] = block[:, -self._capacity:]
            self._write_at = 0
            self._total_written += n
            return

        end = self._write_at + n
        if end <= self._capacity:
            self._data[:, self._write_at:end] = block
        else:
            split = self._capacity - self._write_at
            self._data[:, self._write_at:] = block[:, :split]
            self._data[:, :n - split] = block[:, split:]
        self._write_at = end % self._capacity
        self._total_written += n

    def last(self, n_samples: int) -> np.ndarray:
        """The most recent n_samples, oldest first."""
        available = min(self._total_written, self._capacity)
        n = min(n_samples, available)
        if n == 0:
            return np.zeros((self._data.shape[0], 0))
        start = (self._write_at - n) % self._capacity
        if start + n <= self._capacity:
            return self._data[:, start:start + n].copy()
        split = self._capacity - start
        return np.hstack([self._data[:, start:], self._data[:, :n - split]])

    def slice(self, first_index: int, last_index: int) -> np.ndarray:
        """Samples between two absolute indices, clipped to what is held.

        Returns an empty array if the requested stretch has already been
        overwritten - silently substituting adjacent data would be worse.
        """
        first = max(first_index, self.oldest_index)
        last = min(last_index, self._total_written)
        if last <= first:
            return np.zeros((self._data.shape[0], 0))
        return self.last(self._total_written - first)[:, :last - first]


class SensorAcquisition:
    """Runs a DAQDevice on a worker thread and publishes what it reads."""

    JOIN_TIMEOUT_S = 2.0
    READ_HISTORY = 200          # rolling window for the timing statistics
    # A blocking read waits for the block it returns, so it takes about one
    # chunk duration even when everything is healthy. Only a read
    # substantially longer than that says anything, hence the margin;
    # without it half of all reads would be flagged by ordinary jitter.
    SLOW_READ_MARGIN = 1.5

    def __init__(self, device: DAQDevice,
                 position_source: Optional[PositionSource] = None,
                 ring_seconds: float = 30.0,
                 chunk_samples: Optional[int] = None,
                 position_history: int = 2000):
        self._device = device
        self._position_source = position_source
        self._ring_seconds = ring_seconds
        self._chunk_samples = chunk_samples

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._ring: Optional[RingBuffer] = None
        self._latest: Optional[np.ndarray] = None      # one sample, all channels
        self._positions: Deque[PositionSample] = deque(maxlen=position_history)

        self._sample_rate_hz = 0.0
        self._stream_t0 = 0.0
        self._chunks_read = 0
        self._samples_read = 0
        self._slow_reads = 0
        self._read_durations: Deque[float] = deque(maxlen=self.READ_HISTORY)
        self._last_error: Optional[str] = None

    # --- lifecycle ---------------------------------------------------------

    @property
    def channels(self) -> Tuple[str, ...]:
        return self._device.channels

    @property
    def sample_rate_hz(self) -> float:
        return self._sample_rate_hz

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, sample_rate_hz: float) -> None:
        if self.is_running:
            raise RuntimeError("acquisition already running")

        self._device.start(sample_rate_hz)
        self._sample_rate_hz = sample_rate_hz
        if self._chunk_samples is None:
            self._chunk_samples = default_chunk_samples(sample_rate_hz)

        capacity = max(self._chunk_samples,
                       int(sample_rate_hz * self._ring_seconds))
        with self._lock:
            self._ring = RingBuffer(len(self._device.channels), capacity)
            self._latest = None
            self._positions.clear()
            self._chunks_read = 0
            self._samples_read = 0
            self._slow_reads = 0
            self._read_durations.clear()
            self._last_error = None
            self._stream_t0 = 0.0

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="evexp-acquisition", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: Optional[float] = None) -> None:
        """Ask the worker to finish, wait for it, then release the device."""
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout_s or self.JOIN_TIMEOUT_S)
        try:
            self._device.stop()
        except Exception as exc:                        # noqa: BLE001
            with self._lock:
                self._last_error = f"stop failed: {exc}"

    def __enter__(self) -> "SensorAcquisition":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # --- worker ------------------------------------------------------------

    def _run(self) -> None:
        chunk_samples = self._chunk_samples or 1
        chunk_duration = chunk_samples / self._sample_rate_hz

        while not self._stop.is_set():
            started = time.perf_counter()
            try:
                chunk = self._device.read_chunk(chunk_samples)
            except Exception as exc:                    # noqa: BLE001
                with self._lock:
                    self._last_error = f"read failed: {exc}"
                break

            position = (self._position_source.read()
                        if self._position_source is not None else None)
            elapsed = time.perf_counter() - started
            self._publish(chunk, position, elapsed, chunk_duration)

    def _publish(self, chunk: SensorChunk, position: Optional[PositionSample],
                 read_s: float, chunk_duration_s: float) -> None:
        with self._lock:
            if self._ring is None:
                return
            if self._chunks_read == 0:
                self._stream_t0 = chunk.t0
            self._ring.write(chunk.data)
            self._latest = chunk.data[:, -1].copy()
            if position is not None:
                self._positions.append(position)

            self._chunks_read += 1
            self._samples_read += chunk.n_samples
            self._read_durations.append(read_s)
            # Losing ground to the device. One is noise; a rising count is
            # the warning that arrives before a driver-side overflow does.
            if read_s > chunk_duration_s * self.SLOW_READ_MARGIN:
                self._slow_reads += 1

    # --- readers (called from the Qt thread) -------------------------------

    def latest(self) -> Optional[dict]:
        """Most recent sample on every channel, or None before the first read."""
        with self._lock:
            if self._latest is None:
                return None
            return {name: float(value)
                    for name, value in zip(self._device.channels, self._latest)}

    def latest_channel(self, name: str) -> Optional[float]:
        with self._lock:
            if self._latest is None:
                return None
            try:
                return float(self._latest[self._device.channels.index(name)])
            except ValueError:
                raise KeyError(f"no channel {name!r}") from None

    def recent(self, seconds: float) -> np.ndarray:
        """The last few seconds of every channel, oldest sample first."""
        with self._lock:
            if self._ring is None:
                return np.zeros((len(self._device.channels), 0))
            return self._ring.last(int(seconds * self._sample_rate_hz))

    def window(self, t_start: float, t_end: float) -> np.ndarray:
        """Samples acquired between two host timestamps.

        Returns an empty array if that stretch has already scrolled out of
        the buffer, rather than the nearest data it still has.
        """
        with self._lock:
            if self._ring is None or self._sample_rate_hz <= 0:
                return np.zeros((len(self._device.channels), 0))
            first = int((t_start - self._stream_t0) * self._sample_rate_hz)
            last = int((t_end - self._stream_t0) * self._sample_rate_hz)
            return self._ring.slice(max(0, first), max(0, last))

    def recent_positions(self) -> List[PositionSample]:
        with self._lock:
            return list(self._positions)

    def stats(self) -> AcquisitionStats:
        with self._lock:
            durations = list(self._read_durations)
            elapsed = (time.perf_counter() - self._stream_t0
                       if self._chunks_read else 0.0)
            expected = elapsed * self._sample_rate_hz
            lag_samples = expected - self._samples_read
            return AcquisitionStats(
                running=self.is_running,
                chunks_read=self._chunks_read,
                samples_read=self._samples_read,
                slow_reads=self._slow_reads,
                mean_read_s=(sum(durations) / len(durations)) if durations else 0.0,
                max_read_s=max(durations) if durations else 0.0,
                elapsed_s=elapsed,
                lag_s=(lag_samples / self._sample_rate_hz
                       if self._sample_rate_hz else 0.0),
                last_error=self._last_error,
            )