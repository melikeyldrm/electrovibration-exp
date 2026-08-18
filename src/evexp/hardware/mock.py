"""Stand-ins for hardware that is absent, unfinished, or inconvenient.

These exist so the acquisition path, the trial loop and the UI can all be
exercised end to end without a DAQ card, an amplifier or a force sensor.
They imitate the real interfaces closely enough that swapping the real
implementations in is a change of constructor, not of calling code.
"""

import math
import random
import time
from typing import Optional, Sequence, Tuple

import numpy as np

from evexp.hardware.base import (DAQDevice, SensorChunk, SensorSample,
                                 StimulusOutput)

# Twelve strain-gauge channels: two ATI Nano17s, FS1 on ai0..ai5 and FS2 on
# ai6..ai11 (see nidaq.DEFAULT_CHANNEL_MAP).
DEFAULT_GAUGE_CHANNELS: Tuple[str, ...] = (
    tuple(f"fs1_gauge{i}" for i in range(6))
    + tuple(f"fs2_gauge{i}" for i in range(6))
)


class MockDAQDevice(DAQDevice):
    """Simulated acquisition, block-oriented like the real thing.

    Two behaviours worth knowing about:

    Timing is honest. read_chunk() sleeps until the block it is returning
    would actually have been acquired, so a read loop against this device
    runs at the same rate it will against the card. Without that, threading
    and buffering bugs stay hidden until the hardware arrives, which is the
    worst moment to find them.

    Signals are simple. A slow drift plus noise on each gauge, with a
    load-dependent term so that setting the applied voltage visibly changes
    the data. It is not a model of a fingertip - it is a signal with the
    right shape, rate and dimensionality for the code downstream.
    """

    def __init__(self, channels: Optional[Sequence[str]] = None,
                 true_threshold: float = 50.0, slope: float = 0.15,
                 noise_std: float = 0.02, realtime: bool = True,
                 seed: Optional[int] = None):
        self._channels: Tuple[str, ...] = tuple(channels or DEFAULT_GAUGE_CHANNELS)
        self.true_threshold = true_threshold
        self.slope = slope
        self.noise_std = noise_std
        self.realtime = realtime
        self._rng = np.random.default_rng(seed)
        self._pyrng = random.Random(seed)

        self._applied_voltage: float = 0.0
        self._running = False
        self._sample_rate_hz: float = 1000.0
        self._samples_produced = 0
        self._t_start = 0.0

    # --- DAQDevice ---------------------------------------------------------

    @property
    def channels(self) -> Tuple[str, ...]:
        return self._channels

    @property
    def sample_rate_hz(self) -> float:
        return self._sample_rate_hz

    def start(self, sample_rate_hz: float) -> None:
        if sample_rate_hz <= 0:
            raise ValueError(
                f"sample_rate_hz must be positive, got {sample_rate_hz!r}")
        self._sample_rate_hz = float(sample_rate_hz)
        self._samples_produced = 0
        self._t_start = time.perf_counter()
        self._running = True

    def read_chunk(self, n_samples: int,
                   timeout_s: float = 10.0) -> SensorChunk:
        if not self._running:
            raise RuntimeError("Device must be started before reading.")
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples!r}")

        first_index = self._samples_produced
        t0 = self._t_start + first_index / self._sample_rate_hz

        if self.realtime:
            # Wait until the last sample of this block would have been
            # acquired, so callers see the real cadence.
            ready_at = self._t_start + (first_index + n_samples) / self._sample_rate_hz
            delay = ready_at - time.perf_counter()
            if delay > timeout_s:
                raise TimeoutError(
                    f"chunk of {n_samples} samples needs {delay:.3f} s, "
                    f"longer than the {timeout_s:.3f} s timeout"
                )
            if delay > 0:
                time.sleep(delay)

        data = self._synthesise(first_index, n_samples)
        self._samples_produced += n_samples
        return SensorChunk(
            t0=t0,
            sample_rate_hz=self._sample_rate_hz,
            channels=self._channels,
            data=data,
        )

    def stop(self) -> None:
        self._running = False

    # --- simulated signal --------------------------------------------------

    def _synthesise(self, first_index: int, n_samples: int) -> np.ndarray:
        t = (first_index + np.arange(n_samples)) / self._sample_rate_hz
        n_channels = len(self._channels)
        data = np.empty((n_channels, n_samples))

        # A distinct slow component per channel keeps the channels
        # distinguishable, so a wiring or ordering mistake downstream shows
        # up as obviously wrong rather than as plausible noise.
        for i in range(n_channels):
            drift = 0.05 * math.sin(0.3 + 0.7 * i) * np.sin(2 * np.pi * (0.2 + 0.05 * i) * t)
            data[i] = 0.3 + drift
        data += self._applied_voltage * 0.01
        data += self._rng.normal(0.0, self.noise_std, size=data.shape)
        return data

    # --- stimulus-side helpers used by the headless runner -----------------

    def set_applied_voltage(self, voltage: float) -> None:
        """Update the simulated stimulation voltage."""
        self._applied_voltage = voltage

    def read(self) -> SensorSample:
        """One sample per channel.

        Overridden to stay non-blocking: the base implementation reads a
        one-sample block, which in realtime mode would wait for it.
        """
        if not self._running:
            raise RuntimeError("Device must be started before reading.")
        return SensorSample(
            timestamp=time.time(),
            values={name: 0.3 + self._pyrng.gauss(0, self.noise_std)
                    for name in self._channels},
        )

    def detection_probability(self) -> float:
        """Return simulated detection probability."""
        p_detect = 1 / (1 + math.exp(
            -self.slope * (self._applied_voltage - self.true_threshold)))
        return 0.5 + 0.5 * p_detect


class MockStimulusOutput(StimulusOutput):
    """Stand-in for the DAQ output -> amplifier -> touchscreen chain.

    Records every on/off transition, so a simulated run can be checked for
    the correct number of activations per trial without any hardware.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._amplitude_v = 0.0
        self._active = False
        self.events: list[tuple[float, str, float]] = []

    def set_amplitude(self, volts: float) -> None:
        """Set target stimulus voltage amplitude."""
        self._amplitude_v = volts

    def stimulus_on(self) -> None:
        self._active = True
        self._record("on")

    def stimulus_off(self) -> None:
        # Calling this while already off is harmless; it keeps the caller simple.
        was_active = self._active
        self._active = False
        if was_active:
            self._record("off")

    @property
    def is_active(self) -> bool:
        """Check if stimulus is currently active. read-only"""
        return self._active

    @property
    def amplitude_v(self) -> float:
        """Get current amplitude in volts. read-only"""
        return self._amplitude_v

    def _record(self, kind: str) -> None:
        """Record event timestamp, action, and voltage amplitude."""
        self.events.append((time.time(), kind, self._amplitude_v))
        if self.verbose:
            print(f"[stimulus] {kind:<3s} {self._amplitude_v:6.2f} V")