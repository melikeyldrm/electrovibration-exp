"""Raw per-trial signal writer: fx/fy/fz, commanded voltage, and speed, all
held to the 10 kHz acquisition grid, one HDF5 group per trial.

Kept separate from hdf5_writer.py (the trial-summary file written once at
session end): this one is meant to be appended to as each trial finishes,
since SensorAcquisition's ring buffer only holds ~30 s and a trial cut out
of it too late is gone for good.
"""

from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np

from evexp.hardware.force import ForceCalibration
from evexp.hardware.position import PositionSample
from evexp.processing.signal import hold_to_grid


def _speeds_from_positions(positions: List[PositionSample]):
    """Finite-difference speed (mm/s) between consecutive samples.

    Returns (times, speeds), one shorter than positions: a speed needs two
    samples, so the first position in a trial has none to pair with.
    """
    if len(positions) < 2:
        return np.array([]), np.array([])
    times = np.array([p.t for p in positions])
    xs = np.array([p.x_mm for p in positions])
    dt = np.diff(times)
    dx = np.diff(xs)
    with np.errstate(divide="ignore", invalid="ignore"):
        speeds = np.where(dt > 0, dx / dt, 0.0)
    return times[1:], speeds


class RawSessionWriter:
    """Appends one HDF5 group per trial to a single session file."""

    def __init__(self, output_path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._opened = False

    def _ensure_open(self) -> None:
        # Created on first append, not in __init__: a writer that never
        # receives a trial should not leave an empty file behind that looks
        # like a completed, empty session.
        if not self._opened:
            with h5py.File(self.output_path, "w"):
                pass
            self._opened = True

    def append_trial(self, trial_index: int, gauge_chunk: np.ndarray,
                      calibration: ForceCalibration, sample_rate_hz: float,
                      t0: float, commanded_voltage: float,
                      positions: Optional[List[PositionSample]] = None
                      ) -> None:
        """Write one trial's raw signals as /trials/trial_<index>.

        gauge_chunk: (6, n_samples) raw gauge voltages, already cut from the
            acquisition ring buffer for this trial's time window.
        t0: host timestamp of gauge_chunk's first sample - positions (which
            carry their own timestamps) are aligned against it.
        positions: position samples spanning the trial, at whatever rate the
            position source produces (Neonode: up to 200 Hz). Speed is
            derived from consecutive samples and held to gauge_chunk's
            sample grid, per Umut: "aynı değeri bir süre boyunca tekrar
            yazabilirsin".
        """
        if gauge_chunk.ndim != 2 or gauge_chunk.shape[0] != 6:
            raise ValueError(
                f"gauge_chunk must be (6, n_samples), got {gauge_chunk.shape}")

        n_samples = gauge_chunk.shape[1]
        target_times = t0 + np.arange(n_samples) / sample_rate_hz

        forces = calibration.forces_in_screen_frame(gauge_chunk)  # (3, n)

        if positions:
            pos_times, pos_speeds = _speeds_from_positions(positions)
            speed_held = hold_to_grid(pos_times, pos_speeds, target_times)
        else:
            speed_held = np.full(n_samples, np.nan)

        voltage_held = np.full(n_samples, commanded_voltage, dtype=float)

        self._ensure_open()
        with h5py.File(self.output_path, "a") as f:
            group = f.create_group(f"trials/trial_{trial_index:04d}")
            # Cast explicitly: sample_rate_hz in particular often arrives as
            # a Python/YAML int (e.g. "sample_rate_hz: 10000" with no
            # decimal point), which h5py would otherwise store as an HDF5
            # integer attribute. MATLAB's h5readatt then returns an integer
            # class, and MATLAB refuses to divide a double array by it
            # ("Integers can only be combined with integers of the same
            # class, or scalar doubles") - a division that works fine on the
            # Python/analysis side, so this was invisible until read in
            # MATLAB. Always writing double here avoids the class mismatch
            # regardless of what type the caller passed in.
            group.attrs["t0"] = float(t0)
            group.attrs["sample_rate_hz"] = float(sample_rate_hz)
            group.attrs["commanded_voltage_v"] = float(commanded_voltage)
            group.attrs["calibration_placeholder"] = calibration.is_placeholder
            group.create_dataset("fx", data=forces[0])
            group.create_dataset("fy", data=forces[1])
            group.create_dataset("fz", data=forces[2])
            group.create_dataset("voltage", data=voltage_held)
            group.create_dataset("speed_mm_s", data=speed_held)
            # TODO (pending Umut): a "current" channel belongs here too,
            # once the amplifier-side measurement it comes from is decided.