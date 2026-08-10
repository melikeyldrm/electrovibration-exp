"""Development-only PositionSource/ForceSource implementations.

Not backed by real hardware: mouse-driven or synthetic. Safe to delete once
the Neonode and ATI Nano17 are fully wired up and calibrated - nothing else
in the codebase depends on these classes directly, only on the
PositionSource / ForceSource interfaces they implement.
"""

import math
import random
import threading
import time
from typing import Optional

from evexp.hardware.force import ForceSource
from evexp.hardware.position import PositionSample, PositionSource


class ManualPositionSource(PositionSource):
    """Position driven by whatever the UI pushes in - currently the mouse.

    Written from the Qt thread and read from the acquisition thread, so the
    stored sample is guarded by a lock held only long enough to swap a
    reference.
    """

    def __init__(self, travel_mm: float = 100.0):
        self.travel_mm = travel_mm
        self._lock = threading.Lock()
        self._latest: Optional[PositionSample] = None

    def push(self, x_mm: float) -> None:
        """Called by the UI when the pointer moves over the track."""
        clamped = min(self.travel_mm, max(0.0, x_mm))
        sample = PositionSample(t=time.perf_counter(), x_mm=clamped)
        with self._lock:
            self._latest = sample

    def clear(self) -> None:
        """Called when the pointer leaves the track: no finger detected."""
        with self._lock:
            self._latest = None

    def read(self) -> Optional[PositionSample]:
        with self._lock:
            return self._latest


class SimulatedForceSource(ForceSource):
    """No hardware: drifts around target_n with noise, for dev/testing."""

    def __init__(self, target_n: float = 1.0, drift_n: float = 0.35,
                 drift_period_s: float = 9.0, noise_n: float = 0.02,
                 seed: Optional[int] = None):
        self.target_n = target_n
        self.drift_n = drift_n
        self.drift_period_s = drift_period_s
        self.noise_n = noise_n
        self._rng = random.Random(seed)
        self._t0 = time.perf_counter()

    def read_normal_force(self) -> Optional[float]:
        elapsed = time.perf_counter() - self._t0
        drift = self.drift_n * math.sin(2 * math.pi * elapsed / self.drift_period_s)
        return self.target_n + drift + self._rng.gauss(0.0, self.noise_n)


class ManualForceSource(ForceSource):
    """Force pushed in from the UI (e.g. pointer Y), for dev/testing."""

    def __init__(self, initial_n: Optional[float] = None):
        self._lock = threading.Lock()
        self._value = initial_n

    def push(self, force_n: float) -> None:
        with self._lock:
            self._value = float(force_n)

    def clear(self) -> None:
        with self._lock:
            self._value = None

    def read_normal_force(self) -> Optional[float]:
        with self._lock:
            return self._value