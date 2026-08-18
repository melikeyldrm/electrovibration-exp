import csv

import numpy as np
import pytest

from evexp.data.raw_csv_writer import RawTrialWriter
from evexp.hardware.force import DualForceCalibration, ForceCalibration
from evexp.hardware.position import PositionSample


@pytest.fixture
def placeholder_calibration():
    return DualForceCalibration.placeholder()


def _read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_creates_file_only_after_write_trial(tmp_path, placeholder_calibration):
    writer = RawTrialWriter(tmp_path, participant_id="003", speed_mm_s=50.0,
                             stamp="20260101_000000")
    path = writer._trial_path(0)
    assert not path.exists()

    writer.write_trial(0, np.zeros((12, 10)), np.zeros((12, 10)),
                        placeholder_calibration, sample_rate_hz=10000.0,
                        t0_interval1=0.0, t0_interval2=1.0,
                        voltage_interval1=0.0, voltage_interval2=1.5,
                        stimulus_interval=2)
    assert path.exists()


def test_filename_format(tmp_path, placeholder_calibration):
    writer = RawTrialWriter(tmp_path, participant_id="003", speed_mm_s=50.0,
                             stamp="20260101_000000")
    path = writer._trial_path(7)
    assert path.name == "p003_s50_trial007_20260101_000000.csv"
    assert path.parent.name == "p003"


def test_columns_and_row_count(tmp_path, placeholder_calibration):
    writer = RawTrialWriter(tmp_path, participant_id="003", speed_mm_s=50.0,
                             stamp="stamp")
    n1, n2 = 100, 150
    writer.write_trial(0, np.zeros((12, n1)), np.zeros((12, n2)),
                        placeholder_calibration, sample_rate_hz=10000.0,
                        t0_interval1=0.0, t0_interval2=1.0,
                        voltage_interval1=1.0, voltage_interval2=0.0,
                        stimulus_interval=1)
    rows = _read_rows(writer._trial_path(0))
    assert list(rows[0].keys()) == [
        "interval", "time_s", "fx1", "fy1", "fz1", "fx2", "fy2", "fz2",
        "fx", "fy", "fz", "voltage_v", "current_a", "actuation", "speed_mm_s",
    ]
    assert len(rows) == n1 + n2
    assert sum(1 for r in rows if r["interval"] == "1") == n1
    assert sum(1 for r in rows if r["interval"] == "2") == n2


def test_actuation_marks_stimulus_interval(tmp_path, placeholder_calibration):
    writer = RawTrialWriter(tmp_path, participant_id="003", speed_mm_s=50.0,
                             stamp="stamp")
    writer.write_trial(0, np.zeros((12, 10)), np.zeros((12, 10)),
                        placeholder_calibration, sample_rate_hz=10000.0,
                        t0_interval1=0.0, t0_interval2=1.0,
                        voltage_interval1=0.0, voltage_interval2=1.5,
                        stimulus_interval=2)
    rows = _read_rows(writer._trial_path(0))
    interval1_rows = [r for r in rows if r["interval"] == "1"]
    interval2_rows = [r for r in rows if r["interval"] == "2"]
    assert all(r["actuation"] == "0" for r in interval1_rows)
    assert all(r["actuation"] == "1" for r in interval2_rows)


def test_voltage_channel_holds_commanded_value_per_interval(
        tmp_path, placeholder_calibration):
    writer = RawTrialWriter(tmp_path, participant_id="003", speed_mm_s=50.0,
                             stamp="stamp")
    writer.write_trial(0, np.zeros((12, 10)), np.zeros((12, 10)),
                        placeholder_calibration, sample_rate_hz=10000.0,
                        t0_interval1=0.0, t0_interval2=1.0,
                        voltage_interval1=0.0, voltage_interval2=1.75,
                        stimulus_interval=2)
    rows = _read_rows(writer._trial_path(0))
    v1 = {float(r["voltage_v"]) for r in rows if r["interval"] == "1"}
    v2 = {float(r["voltage_v"]) for r in rows if r["interval"] == "2"}
    assert v1 == {0.0}
    assert len(v2) == 1 and v2.pop() == pytest.approx(1.75)


def test_speed_derived_and_held_from_positions(tmp_path, placeholder_calibration):
    writer = RawTrialWriter(tmp_path, participant_id="003", speed_mm_s=50.0,
                             stamp="stamp")
    n = 100  # 100 samples at 10 kHz over interval 1 -> 0.01 s window
    # Two position samples 0.005 s apart, 1 mm apart -> 200 mm/s, timestamps
    # absolute (t0=0.0 for interval 1).
    positions = [
        PositionSample(t=0.0, x_mm=0.0),
        PositionSample(t=0.005, x_mm=1.0),
    ]
    writer.write_trial(0, np.zeros((12, n)), np.zeros((12, n)),
                        placeholder_calibration, sample_rate_hz=10000.0,
                        t0_interval1=0.0, t0_interval2=1.0,
                        voltage_interval1=0.0, voltage_interval2=0.0,
                        stimulus_interval=1, positions1=positions)
    rows = _read_rows(writer._trial_path(0))
    interval1_speed = [float(r["speed_mm_s"]) for r in rows
                        if r["interval"] == "1"]
    # Before the second position sample arrives -> NaN; after -> 200 mm/s.
    assert all(np.isnan(v) for v in interval1_speed[:50])
    assert all(v == pytest.approx(200.0) for v in interval1_speed[50:])


def test_no_positions_gives_nan_speed(tmp_path, placeholder_calibration):
    writer = RawTrialWriter(tmp_path, participant_id="003", speed_mm_s=50.0,
                             stamp="stamp")
    writer.write_trial(0, np.zeros((12, 20)), np.zeros((12, 20)),
                        placeholder_calibration, sample_rate_hz=10000.0,
                        t0_interval1=0.0, t0_interval2=1.0,
                        voltage_interval1=0.0, voltage_interval2=0.0,
                        stimulus_interval=1)
    rows = _read_rows(writer._trial_path(0))
    assert all(np.isnan(float(r["speed_mm_s"])) for r in rows)


def test_current_defaults_to_nan_when_not_provided(tmp_path, placeholder_calibration):
    writer = RawTrialWriter(tmp_path, participant_id="003", speed_mm_s=50.0,
                             stamp="stamp")
    writer.write_trial(0, np.zeros((12, 5)), np.zeros((12, 5)),
                        placeholder_calibration, sample_rate_hz=10000.0,
                        t0_interval1=0.0, t0_interval2=1.0,
                        voltage_interval1=0.0, voltage_interval2=0.0,
                        stimulus_interval=1)
    rows = _read_rows(writer._trial_path(0))
    assert all(np.isnan(float(r["current_a"])) for r in rows)


def test_current_channel_written_when_provided(tmp_path, placeholder_calibration):
    writer = RawTrialWriter(tmp_path, participant_id="003", speed_mm_s=50.0,
                             stamp="stamp")
    current1 = np.full(5, 0.42)
    writer.write_trial(0, np.zeros((12, 5)), np.zeros((12, 5)),
                        placeholder_calibration, sample_rate_hz=10000.0,
                        t0_interval1=0.0, t0_interval2=1.0,
                        voltage_interval1=0.0, voltage_interval2=0.0,
                        stimulus_interval=1, current1=current1)
    rows = _read_rows(writer._trial_path(0))
    interval1_current = [float(r["current_a"]) for r in rows
                          if r["interval"] == "1"]
    assert interval1_current == pytest.approx([0.42] * 5)


def test_forces_use_the_calibration(tmp_path):
    # A non-identity calibration should visibly change fx/fy/fz vs raw volts.
    # FS1 doubles, FS2 triples, so fz must be the sum of the two - not one
    # sensor's contribution silently standing in for both.
    cal = DualForceCalibration(fs1=ForceCalibration(matrix=np.eye(6) * 2.0),
                               fs2=ForceCalibration(matrix=np.eye(6) * 3.0))
    writer = RawTrialWriter(tmp_path, participant_id="003", speed_mm_s=50.0,
                             stamp="stamp")
    gauge_chunk = np.full((12, 10), 3.0)
    writer.write_trial(0, gauge_chunk, gauge_chunk, cal,
                        sample_rate_hz=10000.0,
                        t0_interval1=0.0, t0_interval2=1.0,
                        voltage_interval1=0.0, voltage_interval2=0.0,
                        stimulus_interval=1)
    rows = _read_rows(writer._trial_path(0))
    assert all(float(r["fz1"]) == pytest.approx(6.0) for r in rows)
    assert all(float(r["fz2"]) == pytest.approx(9.0) for r in rows)
    assert all(float(r["fz"]) == pytest.approx(15.0) for r in rows)


def test_wrong_gauge_shape_raises(tmp_path, placeholder_calibration):
    writer = RawTrialWriter(tmp_path, participant_id="003", speed_mm_s=50.0,
                             stamp="stamp")
    with pytest.raises(ValueError):
        writer.write_trial(0, np.zeros((6, 10)), np.zeros((12, 10)),
                            placeholder_calibration, sample_rate_hz=10000.0,
                            t0_interval1=0.0, t0_interval2=1.0,
                            voltage_interval1=0.0, voltage_interval2=0.0,
                            stimulus_interval=1)


def test_multiple_trials_each_get_their_own_file(tmp_path, placeholder_calibration):
    writer = RawTrialWriter(tmp_path, participant_id="003", speed_mm_s=50.0,
                             stamp="stamp")
    for i in range(3):
        writer.write_trial(i, np.zeros((12, 10)), np.zeros((12, 10)),
                            placeholder_calibration, sample_rate_hz=10000.0,
                            t0_interval1=0.0, t0_interval2=1.0,
                            voltage_interval1=0.0, voltage_interval2=0.0,
                            stimulus_interval=1)
    files = sorted(p.name for p in writer.participant_dir.glob("*.csv"))
    assert files == [
        "p003_s50_trial000_stamp.csv",
        "p003_s50_trial001_stamp.csv",
        "p003_s50_trial002_stamp.csv",
    ]