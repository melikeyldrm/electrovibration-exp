"""Zero-order-hold resampling: stretch a slow signal onto a fast time grid.

Used to align the ~200 Hz position/speed stream with the 10 kHz force/AI
stream for a single HDF5 file with one sample count per trial: each slow
sample is repeated across every fast-grid timestamp until the next slow
sample arrives, per Umut's instruction ("aynı değeri bir süre boyunca tekrar
yazabilirsin").
"""

from typing import Optional

import numpy as np


def hold_to_grid(sample_times: np.ndarray, sample_values: np.ndarray,
                  target_times: np.ndarray,
                  fill_before_first: Optional[float] = np.nan) -> np.ndarray:
    """Zero-order-hold sample_values (at sample_times) onto target_times.

    For each target time t, returns the value of the most recent sample at
    or before t - i.e. the value "holds" until the next slow sample updates
    it, rather than interpolating between them. sample_times must be sorted
    ascending.

    target_times before the first sample have no real value to hold; they
    are filled with fill_before_first (NaN by default, so a gap at the very
    start of a trial is visible rather than silently backfilled with data
    that did not exist yet).
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

    # searchsorted(..., side='right') - 1 gives, for each target time, the
    # index of the last sample at or before it.
    idx = np.searchsorted(sample_times, target_times, side="right") - 1

    held = np.empty(target_times.shape, dtype=float)
    before_first = idx < 0
    held[before_first] = fill_before_first
    held[~before_first] = sample_values[idx[~before_first]]
    return held