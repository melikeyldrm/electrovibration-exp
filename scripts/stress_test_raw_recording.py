"""Stress-test the raw HDF5 recording pipeline over a long staircase run.

Simulates N back-to-back trials with real-timed acquisition (MockDAQDevice
runs at the same 10 kHz cadence a real DAQ card would) and measures:
  - how long each write_trial() call takes (disk I/O)
  - whether the acquisition thread falls behind (lag_s)
  - whether any trial's cut window came back empty (data loss)

No real hardware needed - this exercises the exact same
RecordingController / RawSessionWriter / SensorAcquisition code path a
real session uses, just with a synthetic signal standing in for the DAQ.

Usage:
    python scripts/stress_test_raw_recording.py
    python scripts/stress_test_raw_recording.py --n-trials 80 --interval-s 1.25
"""

import argparse
import time
from pathlib import Path

import numpy as np

from evexp.hardware.mock import MockDAQDevice
from evexp.hardware.acquisition import SensorAcquisition
from evexp.hardware.dev_sources import ManualPositionSource
from evexp.hardware.force import ForceCalibration
from evexp.data.raw_hdf5_writer import RawSessionWriter
from evexp.processing.force_feedback import ForceBands
from evexp.ui.recording_controller import RecordingController


class _FakeResult:
    """Stand-in for psychophysics.trial.TrialResult - only the fields
    RecordingController.write_trial() actually reads."""

    def __init__(self, idx: int):
        self.trial_index = idx
        self.training = False
        self.stimulus_interval = 1 if idx % 2 == 0 else 2
        self.applied_voltage = 1.0
        self.mean_normal_force_n = None
        self.std_normal_force_n = None
        self.force_in_band_fraction = None


def run(n_trials: int, interval_s: float, output_path: str) -> None:
    device = MockDAQDevice(realtime=True)  # honest timing, same cadence as real hardware
    pos_source = ManualPositionSource(travel_mm=100.0)
    acq = SensorAcquisition(device=device, position_source=pos_source, ring_seconds=30.0)
    acq.start(sample_rate_hz=10000.0)

    calibration = ForceCalibration.placeholder()
    bands = ForceBands(target_n=1.0, full_scale_n=0.5)
    writer = RawSessionWriter(output_path)
    recording = RecordingController(acq, writer, calibration, bands, log_fn=lambda s: None)

    write_durations = []
    empty_windows = []
    errors = []

    print(f"Running {n_trials} trials, {interval_s:.2f} s per interval "
          f"(~{n_trials * (2 * interval_s + 0.5):.0f} s total)...")
    overall_start = time.perf_counter()

    for i in range(n_trials):
        recording.mark_interval1_start()
        time.sleep(interval_s)              # the actual stimulus-carrying interval
        recording.mark_interval1_end()
        time.sleep(0.5)                     # gap between intervals
        recording._interval2_start_perf = time.perf_counter()
        time.sleep(interval_s)
        recording._interval2_end_perf = time.perf_counter()

        result = _FakeResult(i)
        w0 = time.perf_counter()
        try:
            recording.write_trial(result)
        except Exception as e:                          # noqa: BLE001
            errors.append((i, str(e)))
        write_durations.append(time.perf_counter() - w0)

        if (i + 1) % 10 == 0:
            print(f"  ... {i + 1}/{n_trials} trials done")

    overall_elapsed = time.perf_counter() - overall_start
    acq.stop()
    stats = acq.stats()

    print()
    print("=== Results ===")
    print(f"Total wall time: {overall_elapsed:.2f} s")
    print(f"write_trial() duration: mean={np.mean(write_durations)*1000:.2f} ms, "
          f"max={np.max(write_durations)*1000:.2f} ms")
    print(f"Acquisition: {stats.describe()}")
    print(f"Acquisition lag_s: {stats.lag_s:.4f}  "
          f"(should stay near 0 - growing lag means the read loop is falling behind)")
    print(f"Slow reads: {stats.slow_reads}")
    print(f"Errors during write_trial: {len(errors)}")
    for idx, e in errors:
        print(f"  trial {idx}: {e}")
    print()
    print(f"Wrote: {Path(output_path).resolve()}")
    print()
    if not errors and stats.lag_s < 0.5:
        print("PASS: no write errors, no data loss, acquisition kept up.")
    else:
        print("CHECK NEEDED: see errors / lag above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=50,
                         help="number of trials to simulate (default: 50)")
    parser.add_argument("--interval-s", type=float, default=2.0,
                         help="stimulus interval duration in seconds "
                              "(default: 2.0, matching 100mm/50mm/s)")
    parser.add_argument("--output", default="stress_test_raw.h5",
                         help="output HDF5 path (default: stress_test_raw.h5)")
    args = parser.parse_args()
    run(args.n_trials, args.interval_s, args.output)