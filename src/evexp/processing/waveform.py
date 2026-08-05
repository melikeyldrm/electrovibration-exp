"""Stimulus waveform generation.

Pure computation, kept apart from hardware/nidaq.py so it can be unit-tested
without a DAQ card and reused if the output path changes.
"""

import numpy as np


def generate_sine_wave(frequency_hz: float, amplitude_v: float,
                        duration_s: float, sample_rate_hz: float) -> np.ndarray:
    """Generate a sine wave sample buffer, starting at phase 0.

    Starting at phase 0 keeps buffers phase-continuous when concatenated or
    regenerated, provided duration_s covers a whole number of cycles.
    """
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
    """Duration covering an exact number of cycles at frequency_hz.

    Used to size a regeneration buffer so it loops without a phase jump at
    the wrap point.
    """
    if frequency_hz <= 0:
        raise ValueError(
            f"frequency_hz must be positive, got {frequency_hz!r}")
    if n_cycles <= 0:
        raise ValueError(f"n_cycles must be positive, got {n_cycles!r}")
    return n_cycles / frequency_hz