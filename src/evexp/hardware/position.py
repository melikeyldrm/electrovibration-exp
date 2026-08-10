"""Finger-position input, abstracted from its source.

The participant's finger position is needed to draw their marker on the
pacing track. Dev-only sources (mouse-driven, etc.) live in dev_sources.py;
a real NeonodePositionSource implementing PositionSource drops in without
changing anything in the UI.

Mirrors the StimulusOutput / MockStimulusOutput pattern in hardware/base.py.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class PositionSample:
    """A single position reading.

    Position is in millimetres along the travel axis, not pixels, so the
    same numbers are meaningful for both the UI and later analysis (sliding
    speed affects electrovibration perception, so these are worth logging).
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