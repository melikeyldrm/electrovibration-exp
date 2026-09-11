"""Raw per-trial signal writer: one CSV file per trial, one folder per
participant, both intervals stacked in the same file.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np

from evexp.hardware.force import DualForceCalibration
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


class RawTrialWriter:
    """Writes one CSV per trial under <output_dir>/p<participant_id>/.

    Filename: p<participant_id>_s<speed>_trial<index:03d>_<stamp>.csv
    Rows for interval 1 come first, then interval 2. `actuation` is 1 on
    the interval that carried the stimulus, 0 on the other. Both sensors
    are written separately (fx1.. / fx2..) alongside their sum (fx/fy/fz).
    """

    COLUMNS = ("interval,time_s,fx1,fy1,fz1,fx2,fy2,fz2,fx,fy,fz,"
               "voltage_v,current_a,actuation,speed_mm_s")

    def __init__(self, output_dir, participant_id: str, speed_mm_s: float,
                 stamp: str):
        self.participant_dir = Path(output_dir) / f"p{participant_id}"
        self.participant_dir.mkdir(parents=True, exist_ok=True)
        self.participant_id = participant_id
        self.speed_mm_s = speed_mm_s
        # One stamp for the whole session, passed in rather than generated
        # per trial, so every file from the same run shares a suffix.
        self.stamp = stamp

    def _trial_path(self, trial_index: int) -> Path:
        return self.participant_dir / (
            f"p{self.participant_id}_s{self.speed_mm_s:g}_"
            f"trial{trial_index:03d}_{self.stamp}.csv"
        )

    def _interval_rows(
        self, interval_label: int, gauge_chunk: np.ndarray,
        calibration: DualForceCalibration, sample_rate_hz: float, t0: float,
        commanded_voltage: float, actuation: int,
        positions: Optional[List[PositionSample]],
        current_chunk: Optional[np.ndarray],
    ) -> List[str]:
        n = gauge_chunk.shape[1]
        rel_times = np.arange(n) / sample_rate_hz
        abs_times = t0 + rel_times
        forces1, forces2 = calibration.forces_per_sensor(gauge_chunk)  # (3, n)
        forces = forces1 + forces2

        if positions:
            pos_times, pos_speeds = _speeds_from_positions(positions)
            speed = hold_to_grid(pos_times, pos_speeds, abs_times)
        else:
            speed = np.full(n, np.nan)

        # current_a: AI channel not wired yet; NaN until it is.
        current = current_chunk if current_chunk is not None else np.full(n, np.nan)

        return [
            f"{interval_label},{rel_times[i]:.6f},"
            f"{forces1[0, i]:.6f},{forces1[1, i]:.6f},{forces1[2, i]:.6f},"
            f"{forces2[0, i]:.6f},{forces2[1, i]:.6f},{forces2[2, i]:.6f},"
            f"{forces[0, i]:.6f},{forces[1, i]:.6f},{forces[2, i]:.6f},"
            f"{commanded_voltage:.6f},{current[i]:.6f},{actuation},"
            f"{speed[i]:.6f}"
            for i in range(n)
        ]

    def write_trial(
        self, trial_index: int,
        gauge1: np.ndarray, gauge2: np.ndarray,
        calibration: DualForceCalibration, sample_rate_hz: float,
        t0_interval1: float, t0_interval2: float,
        voltage_interval1: float, voltage_interval2: float,
        stimulus_interval: int,
        positions1: Optional[List[PositionSample]] = None,
        positions2: Optional[List[PositionSample]] = None,
        current1: Optional[np.ndarray] = None,
        current2: Optional[np.ndarray] = None,
    ) -> None:
        """Write both intervals of one trial to a single CSV file.

        gauge1/gauge2: (12, n_samples) raw gauge voltages - FS1 on rows
            0-5, FS2 on rows 6-11 - already cut from the acquisition ring
            buffer for each interval's time window.
        t0_interval1/2: host timestamp of each chunk's first sample, used to
            align positions (which carry their own absolute timestamps).
        """
        for label, chunk in ((1, gauge1), (2, gauge2)):
            if chunk.ndim != 2 or chunk.shape[0] != 12:
                raise ValueError(
                    f"interval {label} chunk must be (12, n_samples), "
                    f"got {chunk.shape}")

        rows = [self.COLUMNS]
        rows += self._interval_rows(
            1, gauge1, calibration, sample_rate_hz, t0_interval1,
            voltage_interval1, 1 if stimulus_interval == 1 else 0,
            positions1, current1,
        )
        rows += self._interval_rows(
            2, gauge2, calibration, sample_rate_hz, t0_interval2,
            voltage_interval2, 1 if stimulus_interval == 2 else 0,
            positions2, current2,
        )
        self._trial_path(trial_index).write_text("\n".join(rows) + "\n")