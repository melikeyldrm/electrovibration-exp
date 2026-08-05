"""Hard safety limit on the stimulus signal sent to the DAQ."""

import numpy as np


class SafetyLimitExceeded(Exception):
    """Raised when a signal would exceed the hard voltage limit."""


def check_voltage_limit(signal: np.ndarray, limit_v: float = 2.0) -> None:
    """Raise if any sample in signal exceeds +/- limit_v.

    Rejects rather than clamps: silently clipping a bad signal would hide
    a config or unit error instead of surfacing it.
    """
    peak = float(np.max(np.abs(signal)))
    if peak > limit_v:
        raise SafetyLimitExceeded(
            f"stimulus peak {peak:.4f} V exceeds hard limit ±{limit_v} V"
        )