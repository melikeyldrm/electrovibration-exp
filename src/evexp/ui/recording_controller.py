"""Cuts and writes per-trial raw signal windows from the acquisition ring buffer.

Split out from SessionController because raw recording needs the
acquisition ring buffer and a force calibration, neither of which the rest
of session orchestration (trial progression, force feedback, logging) has
any business knowing about.
"""

import time
from typing import Callable, Optional

from evexp.hardware.acquisition import SensorAcquisition
from evexp.hardware.force import ForceCalibration
from evexp.data.raw_hdf5_writer import RawSessionWriter
from evexp.processing.force_feedback import ForceBands, force_stats_from_samples


class RecordingController:
    """Marks interval boundaries and writes the stimulus-carrying one.

    All three of acquisition/raw_writer/force_calibration are required
    together; missing any one simply disables raw recording (`enabled`
    stays False), rather than half-recording.
    """

    def __init__(
        self,
        acquisition: Optional[SensorAcquisition],
        raw_writer: Optional[RawSessionWriter],
        force_calibration: Optional[ForceCalibration],
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
        # Both intervals' boundaries are tracked because which one carries
        # the stimulus is randomised per trial by Trial2IFC.
        self._interval1_start_perf = 0.0
        self._interval1_end_perf = 0.0
        self._interval2_start_perf = 0.0
        self._interval2_end_perf = 0.0

    def mark_interval1_start(self) -> None:
        self._interval1_start_perf = time.perf_counter()

    def mark_interval1_end(self) -> None:
        self._interval1_end_perf = time.perf_counter()

    def mark_interval2_start(self) -> None:
        self._interval2_start_perf = time.perf_counter()

    def mark_interval2_end(self) -> None:
        self._interval2_end_perf = time.perf_counter()

    def write_trial(self, result) -> None:
        """Cut only the stimulus-carrying interval from the ring buffer.

        No-op if recording is not enabled. Best-effort otherwise: a failure
        here must not stop the trial loop or lose the CSV row that follows,
        so errors are logged and swallowed. An empty window (ring buffer
        overrun, or acquisition never running) is skipped the same way.
        """
        if not self.enabled:
            return
        try:
            if result.stimulus_interval == 1:
                t_start, t_end = self._interval1_start_perf, self._interval1_end_perf
            else:
                t_start, t_end = self._interval2_start_perf, self._interval2_end_perf

            block = self._acquisition.window(t_start, t_end)
            if block.shape[1] == 0:
                self._log(
                    "  [raw] skipped: empty window (ring buffer overrun or "
                    "acquisition not running)"
                )
                return

            channels = self._acquisition.channels
            gauge_indices = [channels.index(f"gauge{i}") for i in range(6)]
            gauge_chunk = block[gauge_indices, :]

            # Exact force stats from the same samples as the raw HDF5 write,
            # not the sparse 20 Hz poll SessionController's accumulator uses
            # - keeps the CSV summary consistent with the raw signal.
            if self._force_bands is not None:
                forces_n = self._force_calibration.normal_force(gauge_chunk)
                stats = force_stats_from_samples(forces_n, self._force_bands)
                result.mean_normal_force_n = stats.mean_n
                result.std_normal_force_n = stats.std_n
                result.force_in_band_fraction = stats.in_band_fraction

            positions = [
                p for p in self._acquisition.recent_positions()
                if t_start <= p.t <= t_end
            ]

            self._raw_writer.append_trial(
                trial_index=result.trial_index,
                gauge_chunk=gauge_chunk,
                calibration=self._force_calibration,
                sample_rate_hz=self._acquisition.sample_rate_hz,
                t0=t_start,
                commanded_voltage=result.applied_voltage,
                positions=positions,
            )
            self._log(
                f"  [raw] wrote trial {result.trial_index} "
                f"(stimulus interval {result.stimulus_interval}): "
                f"{gauge_chunk.shape[1]} samples, {len(positions)} positions"
            )
        except Exception as exc:                       # noqa: BLE001
            self._log(f"  [raw] FAILED to write trial: {exc}")