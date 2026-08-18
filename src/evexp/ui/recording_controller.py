"""Cuts and writes per-trial raw signal windows from the acquisition ring buffer.

Split out from SessionController because raw recording needs the
acquisition ring buffer and a force calibration, neither of which the rest
of session orchestration (trial progression, force feedback, logging) has
any business knowing about.

Each interval is cut from the ring buffer as soon as it is safely complete,
not when the participant responds: the buffer holds ~30 s, and a participant
is free to take as long as they like to answer, so a response-triggered cut
silently loses interval 1 on slow trials.
"""

import time
from typing import Callable, List, NamedTuple, Optional

import numpy as np

from evexp.hardware.acquisition import SensorAcquisition
from evexp.hardware.force import (ALL_GAUGE_CHANNELS, DualForceCalibration)
from evexp.data.raw_csv_writer import RawTrialWriter
from evexp.processing.force_feedback import ForceBands, force_stats_from_samples


class IntervalCapture(NamedTuple):
    """One interval's signals, already cut out of the ring buffer."""
    gauge: np.ndarray
    current: Optional[np.ndarray]
    positions: List
    t0: float


class RecordingController:
    """Captures each interval as it completes, then writes both to one CSV.

    All three of acquisition/raw_writer/force_calibration are required
    together; missing any one simply disables raw recording (`enabled`
    stays False), rather than half-recording.
    """

    def __init__(
        self,
        acquisition: Optional[SensorAcquisition],
        raw_writer: Optional[RawTrialWriter],
        force_calibration: Optional[DualForceCalibration],
        force_bands: Optional[ForceBands] = None,
        log_fn: Callable[[str], None] = print,
    ):
        self._acquisition = acquisition
        self._raw_writer = raw_writer
        self._force_calibration = force_calibration
        self._force_bands = force_bands
        self._log = log_fn
        self.enabled = (
            acquisition is not None and raw_writer is not None
            and force_calibration is not None
        )
        self._interval1_start_perf = 0.0
        self._interval1_end_perf = 0.0
        self._interval2_start_perf = 0.0
        self._interval2_end_perf = 0.0
        self._capture1: Optional[IntervalCapture] = None
        self._capture2: Optional[IntervalCapture] = None

    def mark_interval1_start(self) -> None:
        # New trial begins here: drop any captures left from the previous
        # one, so a failed capture cannot be written twice.
        self._capture1 = None
        self._capture2 = None
        self._interval1_start_perf = time.perf_counter()

    def mark_interval1_end(self) -> None:
        self._interval1_end_perf = time.perf_counter()

    def mark_interval2_start(self) -> None:
        self._interval2_start_perf = time.perf_counter()
        # Interval 1 is cut here rather than at its own end mark: the gap
        # and pre-interval wait have passed, so every sample of it has
        # certainly reached the ring buffer by now.
        self._capture1 = self._capture(
            self._interval1_start_perf, self._interval1_end_perf, "interval 1")

    def mark_interval2_end(self) -> None:
        self._interval2_end_perf = time.perf_counter()

    def capture_interval2(self) -> None:
        """Cut interval 2. Call shortly after mark_interval2_end().

        Not called from mark_interval2_end() directly: the acquisition
        thread reads in ~50 ms chunks, so the tail of the interval has not
        reached the ring buffer at the instant the interval ends. A short
        delay costs nothing and keeps the cut independent of response time.
        """
        self._capture2 = self._capture(
            self._interval2_start_perf, self._interval2_end_perf, "interval 2")

    def _capture(self, t_start: float, t_end: float,
                 label: str) -> Optional[IntervalCapture]:
        if not self.enabled:
            return None
        try:
            block = self._acquisition.window(t_start, t_end)
            if block.shape[1] == 0:
                self._log(f"  [raw] {label}: empty window (ring buffer "
                          "overrun or acquisition not running)")
                return None
            channels = self._acquisition.channels
            # Both sensors' gauges, FS1 first - the order DualForceCalibration
            # expects when it splits the block back into two.
            gauge_indices = [channels.index(name) for name in ALL_GAUGE_CHANNELS]
            current = (block[channels.index("current"), :]
                       if "current" in channels else None)
            positions = [p for p in self._acquisition.recent_positions()
                         if t_start <= p.t <= t_end]
            return IntervalCapture(
                gauge=block[gauge_indices, :],
                current=current,
                positions=positions,
                t0=t_start,
            )
        except Exception as exc:                       # noqa: BLE001
            self._log(f"  [raw] {label}: FAILED to capture: {exc}")
            return None

    def write_trial(self, result) -> None:
        """Write both captured intervals to one CSV file.

        No-op if recording is not enabled. Best-effort otherwise: a failure
        here must not stop the trial loop or lose the CSV row that follows,
        so errors are logged and swallowed.
        """
        if not self.enabled:
            return
        if self._capture1 is None or self._capture2 is None:
            missing = [n for n, c in (("1", self._capture1),
                                      ("2", self._capture2)) if c is None]
            self._log(f"  [raw] skipped trial {result.trial_index}: "
                      f"interval {'/'.join(missing)} not captured")
            return
        try:
            # Force stats summarise the stimulus-carrying interval only -
            # that is the one whose contact quality the trial depends on.
            # Exact stats from the same samples as the raw write, not the
            # sparse 20 Hz poll SessionController's accumulator uses.
            if self._force_bands is not None:
                stim = (self._capture1 if result.stimulus_interval == 1
                        else self._capture2)
                forces_n = self._force_calibration.normal_force(stim.gauge)
                stats = force_stats_from_samples(forces_n, self._force_bands)
                result.mean_normal_force_n = stats.mean_n
                result.std_normal_force_n = stats.std_n
                result.force_in_band_fraction = stats.in_band_fraction

            v = result.applied_voltage
            self._raw_writer.write_trial(
                trial_index=result.trial_index,
                gauge1=self._capture1.gauge,
                gauge2=self._capture2.gauge,
                calibration=self._force_calibration,
                sample_rate_hz=self._acquisition.sample_rate_hz,
                t0_interval1=self._capture1.t0,
                t0_interval2=self._capture2.t0,
                voltage_interval1=v if result.stimulus_interval == 1 else 0.0,
                voltage_interval2=v if result.stimulus_interval == 2 else 0.0,
                stimulus_interval=result.stimulus_interval,
                positions1=self._capture1.positions,
                positions2=self._capture2.positions,
                current1=self._capture1.current,
                current2=self._capture2.current,
            )
            self._log(
                f"  [raw] wrote trial {result.trial_index}: "
                f"{self._capture1.gauge.shape[1]} + "
                f"{self._capture2.gauge.shape[1]} samples "
                f"(stimulus interval {result.stimulus_interval})"
            )
        except Exception as exc:                       # noqa: BLE001
            self._log(f"  [raw] FAILED to write trial: {exc}")