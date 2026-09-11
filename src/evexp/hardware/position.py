"""Finger-position input, abstracted from its source."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class PositionSample:
    """A single position reading, in millimetres along the travel axis."""
    t: float          # seconds, time.perf_counter() based
    x_mm: float       # position along the tangential travel axis
    y_mm: float = 0.0


class PositionSource(ABC):
    """Provides the participant's current finger position."""

    @abstractmethod
    def read(self) -> Optional[PositionSample]:
        """Return the latest sample, or None if no finger is detected.

        Must not block: it is called both from a UI timer and from the
        acquisition thread.
        """