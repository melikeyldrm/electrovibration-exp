"""Finger-position input, abstracted from its source.

The participant's finger position is needed to draw their marker on the
pacing track. Right now no working position sensor is available, so the
position is taken from the mouse; when the Neonode zForce touch sensor is
wired up, a NeonodePositionSource implementing the same interface drops in
and nothing in the UI changes.

Mirrors the StimulusOutput / MockStimulusOutput pattern in hardware/base.py.
"""

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class PositionSample:
    """A single position reading.

    Position is stored in millimetres along the travel axis, not pixels, so
    the same numbers are meaningful for both the UI and later analysis
    (sliding speed is known to affect electrovibration perception, so these
    samples are worth logging, not just displaying).
    """
    t: float          # seconds, time.perf_counter() based
    x_mm: float       # position along the tangential travel axis
    y_mm: float = 0.0 # unused for now; the task is one-dimensional


class PositionSource(ABC):
    """Provides the participant's current finger position."""

    @abstractmethod
    def read(self) -> Optional[PositionSample]:
        """Return the latest sample, or None if no finger is detected.

        Must not block: it is called both from a UI timer and from the
        acquisition thread.
        """


class ManualPositionSource(PositionSource):
    """Position driven by whatever the UI pushes in - currently the mouse.

    This is a genuine input rather than a canned trajectory: during a demo
    the marker moves because someone actually moves it, which is what makes
    the pacing feedback worth showing at all.

    Written from the Qt thread and read from the acquisition thread, so the
    stored sample is guarded. The lock is held only long enough to swap a
    reference, which is far shorter than the interval between mouse events.
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