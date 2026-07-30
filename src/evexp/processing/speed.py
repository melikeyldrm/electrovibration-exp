"""Finger speed estimation from position samples.

Kept separate from both the UI and the position source: it is pure
computation, so it can be unit-tested without Qt or hardware, and the same
estimator works whether samples come from the mouse or from a real sensor.
"""

from collections import deque
from typing import Deque, Optional

from evexp.hardware.position import PositionSample


class SpeedEstimator:
    """Estimates speed in mm/s from a short history of position samples.

    A finite-difference between two adjacent samples is far too noisy to
    display (at 60 fps the pointer may not move at all between frames, giving
    a speed of zero). Instead the speed is taken over a sliding window, which
    smooths the reading without adding lag the experimenter would notice.
    """
    
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