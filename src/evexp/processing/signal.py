"""Pure signal-math helpers: stimulus waveforms, resampling, speed estimation.

Kept apart from hardware and UI so it can be unit-tested without a DAQ card,
Qt, or a position sensor.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np

from evexp.hardware.position import PositionSample

# --- stimulus waveform generation ------------------------------------------


def generate_sine_wave(frequency_hz: float, amplitude_v: float,
                        duration_s: float, sample_rate_hz: float) -> np.ndarray:
    """Generate a sine wave sample buffer, starting at phase 0."""
    if sample_rate_hz <= 0:
        raise ValueError(
            f"sample_rate_hz must be positive, got {sample_rate_hz!r}")
    if frequency_hz <= 0:
        raise ValueError(
            f"frequency_hz must be positive, got {frequency_hz!r}")
    if duration_s < 0:
        raise ValueError(
            f"duration_s must be non-negative, got {duration_s!r}")

    n_samples = int(round(duration_s * sample_rate_hz))
    t = np.arange(n_samples) / sample_rate_hz
    return amplitude_v * np.sin(2 * np.pi * frequency_hz * t)


def cycles_to_duration_s(frequency_hz: float, n_cycles: int) -> float:
    """Duration covering an exact number of cycles at frequency_hz."""
    if frequency_hz <= 0:
        raise ValueError(
            f"frequency_hz must be positive, got {frequency_hz!r}")
    if n_cycles <= 0:
        raise ValueError(f"n_cycles must be positive, got {n_cycles!r}")
    return n_cycles / frequency_hz


# --- zero-order-hold resampling ---------------------------------------------


def hold_to_grid(sample_times: np.ndarray, sample_values: np.ndarray,
                  target_times: np.ndarray,
                  fill_before_first: Optional[float] = np.nan) -> np.ndarray:
    """Zero-order-hold sample_values (at sample_times) onto target_times.

    For each target time t, returns the value of the most recent sample at
    or before t; sample_times must be sorted ascending. target_times before
    the first sample are filled with fill_before_first (NaN by default).
    """
    sample_times = np.asarray(sample_times, dtype=float)
    sample_values = np.asarray(sample_values, dtype=float)
    target_times = np.asarray(target_times, dtype=float)

    if sample_times.shape != sample_values.shape:
        raise ValueError(
            f"sample_times and sample_values must match in shape, "
            f"got {sample_times.shape} and {sample_values.shape}"
        )
    if sample_times.ndim != 1:
        raise ValueError("sample_times must be 1-D")
    if sample_times.size == 0:
        return np.full(target_times.shape, fill_before_first, dtype=float)
    if np.any(np.diff(sample_times) < 0):
        raise ValueError("sample_times must be sorted ascending")

    idx = np.searchsorted(sample_times, target_times, side="right") - 1

    held = np.empty(target_times.shape, dtype=float)
    before_first = idx < 0
    held[before_first] = fill_before_first
    held[~before_first] = sample_values[idx[~before_first]]
    return held


# --- finger speed estimation -------------------------------------------------


class SpeedEstimator:
    """Estimates speed in mm/s from a short sliding window of position samples."""

    def __init__(self, window_size: int = 8, stale_after_s: float = 0.3):
        self._samples: Deque[PositionSample] = deque(maxlen=window_size)
        self._stale_after_s = stale_after_s

    def add(self, sample: Optional[PositionSample]) -> None:
        """Add a sample, or None if no finger is currently detected."""
        if sample is None:
            self._samples.clear()
            return
        if self._samples and sample.t == self._samples[-1].t:
            return  # same sample polled twice; nothing new to add
        self._samples.append(sample)

    def reset(self) -> None:
        self._samples.clear()

    def speed_mm_s(self, now: Optional[float] = None) -> Optional[float]:
        """Current speed, or None if there isn't enough recent data.

        Returns None rather than 0.0 when data is stale, so the UI can
        distinguish "finger stopped" from "finger lifted".
        """
        if len(self._samples) < 2:
            return None

        first, last = self._samples[0], self._samples[-1]
        if now is not None and (now - last.t) > self._stale_after_s:
            return None

        dt = last.t - first.t
        if dt <= 0:
            return None
        return abs(last.x_mm - first.x_mm) / dt


@dataclass(frozen=True)
class SpeedTrialStats:
    """Speed summary for one trial. mean_mm_s is None if fewer than two
    position samples arrived (distinct from a measured speed of zero)."""
    mean_mm_s: Optional[float]
    n_samples: int


class SpeedTrialAccumulator:
    """Collects position readings over one trial and summarises mean speed."""

    def __init__(self):
        self._prev: Optional[PositionSample] = None
        self._speeds_mm_s: list = []

    def reset(self) -> None:
        self._prev = None
        self._speeds_mm_s.clear()

    def add(self, sample: Optional[PositionSample]) -> None:
        """Record one reading. None means no contact at that instant."""
        if sample is None:
            self._prev = None
            return
        if self._prev is not None and sample.t != self._prev.t:
            dt = sample.t - self._prev.t
            if dt > 0:
                self._speeds_mm_s.append(
                    abs(sample.x_mm - self._prev.x_mm) / dt)
        self._prev = sample

    def stats(self) -> SpeedTrialStats:
        n = len(self._speeds_mm_s)
        if n == 0:
            return SpeedTrialStats(mean_mm_s=None, n_samples=0)
        return SpeedTrialStats(
            mean_mm_s=sum(self._speeds_mm_s) / n, n_samples=n)