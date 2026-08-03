"""Mapping applied force onto the participant's colour feedback.

Kept separate from the widget that draws it for the same reason speed
estimation is kept separate from the widget that shows it: this is pure
computation, testable without Qt, and the exact shape of the response curve
(band edges, how colours blend, how the border scales) is exactly the kind
of thing that gets tuned after watching a real participant use it - it
should be tunable without touching paint code.

The mapping is deliberately monotonic and single-channel in meaning: one
ordered axis (too little -> on target -> too much), not two independent
"good/bad" judgements that happen to share a colour. A participant pressing
either too hard or too softly must be able to tell which, at a glance,
without reading a number.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from evexp.ui import theme


@dataclass(frozen=True)
class ForceBands:
    """The target normal force and how far off it the colour scale reaches.

    Symmetric around the target by design: the protocol has one target
    force, not independent low/high thresholds, so drifting 0.3 N under and
    0.3 N over should look like equal and opposite errors, not different
    amounts of wrong.
    """

    target_n: float
    full_scale_n: float   # distance from target at which the colour is saturated

    def __post_init__(self) -> None:
        if self.full_scale_n <= 0:
            raise ValueError(
                f"full_scale_n must be positive, got {self.full_scale_n!r}")

    def signed_error(self, force_n: float) -> float:
        """Positive when pressing too hard, negative when too light."""
        return force_n - self.target_n

    def normalized_error(self, force_n: float) -> float:
        """Signed error scaled to [-1, 1], clamped at the full-scale distance."""
        raw = self.signed_error(force_n) / self.full_scale_n
        return max(-1.0, min(1.0, raw))


def _lerp_hex(low: str, high: str, fraction: float) -> str:
    """Blend two "#rrggbb" colours. fraction=0 -> low, fraction=1 -> high."""
    fraction = max(0.0, min(1.0, fraction))
    lo = tuple(int(low[i:i + 2], 16) for i in (1, 3, 5))
    hi = tuple(int(high[i:i + 2], 16) for i in (1, 3, 5))
    blended = tuple(round(a + (b - a) * fraction) for a, b in zip(lo, hi))
    return "#{:02x}{:02x}{:02x}".format(*blended)


def force_color(force_n: Optional[float], bands: ForceBands) -> str:
    """Fill colour for the given force: blue (light) - green (on target) - red (heavy).

    None means no contact, which is a distinct state from "zero force" and
    gets its own neutral colour rather than reading as either extreme.
    """
    if force_n is None:
        return theme.FORCE_UNKNOWN

    error = bands.normalized_error(force_n)
    if error >= 0:
        return _lerp_hex(theme.FORCE_TARGET, theme.FORCE_HIGH, error)
    return _lerp_hex(theme.FORCE_TARGET, theme.FORCE_LOW, -error)


def force_border_px(force_n: Optional[float], bands: ForceBands) -> float:
    """Marker border width: thin on target, thick at full-scale error.

    A second channel carrying the same information as the colour, so the
    feedback does not rely on colour discrimination at all - only on
    noticing the marker's outline getting heavier.
    """
    if force_n is None:
        return theme.FORCE_BORDER_MIN_PX

    magnitude = abs(bands.normalized_error(force_n))
    span = theme.FORCE_BORDER_MAX_PX - theme.FORCE_BORDER_MIN_PX
    return theme.FORCE_BORDER_MIN_PX + magnitude * span


class ForceSmoother:
    """Short moving average, to keep the feedback from flickering.

    A raw force signal jittering across a band edge would flip the colour
    back and forth many times a second, which reads as noise rather than
    feedback. Averaging over a short window (tens of milliseconds) removes
    that without adding lag a participant would notice.
    """

    def __init__(self, window_samples: int = 5):
        if window_samples < 1:
            raise ValueError(
                f"window_samples must be at least 1, got {window_samples!r}")
        self._window = window_samples
        self._values: list = []

    def add(self, force_n: Optional[float]) -> Optional[float]:
        """Feed one reading, get back the smoothed value.

        None resets the average immediately rather than being smoothed
        through: "finger lifted" must show up at once, not fade in over the
        window like a change in force would.
        """
        if force_n is None:
            self._values.clear()
            return None
        self._values.append(force_n)
        if len(self._values) > self._window:
            self._values.pop(0)
        return sum(self._values) / len(self._values)

    def reset(self) -> None:
        self._values.clear()