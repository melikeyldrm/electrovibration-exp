import h5py
import numpy as np
import pytest

from evexp.data.raw_hdf5_writer import RawSessionWriter
from evexp.hardware.force import ForceCalibration
from evexp.hardware.position import PositionSample


@pytest.fixture
def placeholder_calibration():
    return ForceCalibration.placeholder()


def test_creates_file_only_after_first_trial(tmp_path):
    path = tmp_path / "raw.h5"
    writer = RawSessionWriter(path)
    assert not path.exists()

    gauge_chunk = np.zeros((6, 100))
    writer.append_trial(0, gauge_chunk, ForceCalibration.placeholder(),
                         sample_rate_hz=10000.0, t0=0.0,
                         commanded_voltage=1.5)
    assert path.exists()


def test_trial_group_shapes_and_names(tmp_path, placeholder_calibration):
    path = tmp_path / "raw.h5"
    writer = RawSessionWriter(path)
    n = 200
    gauge_chunk = np.ones((6, n))
    writer.append_trial(0, gauge_chunk, placeholder_calibration,
                         sample_rate_hz=10000.0, t0=5.0,
                         commanded_voltage=1.2)

    with h5py.File(path, "r") as f:
        group = f["trials/trial_0000"]
        for name in ("fx", "fy", "fz", "voltage", "speed_mm_s"):
            assert group[name].shape == (n,)
        assert group.attrs["t0"] == 5.0
        assert group.attrs["sample_rate_hz"] == 10000.0
        assert group.attrs["commanded_voltage_v"] == pytest.approx(1.2)


def test_voltage_channel_holds_commanded_value(tmp_path, placeholder_calibration):
    path = tmp_path / "raw.h5"
    writer = RawSessionWriter(path)
    writer.append_trial(0, np.zeros((6, 50)), placeholder_calibration,
                         sample_rate_hz=10000.0, t0=0.0,
                         commanded_voltage=1.75)
    with h5py.File(path, "r") as f:
        voltage = f["trials/trial_0000/voltage"][:]
    assert np.all(voltage == 1.75)


def test_speed_derived_and_held_from_positions(tmp_path, placeholder_calibration):
    path = tmp_path / "raw.h5"
    writer = RawSessionWriter(path)
    n = 100  # 100 samples at 10 kHz = 0.01 s
    gauge_chunk = np.zeros((6, n))
    # Two position samples 0.005 s apart, 1 mm apart -> 200 mm/s.
    positions = [
        PositionSample(t=0.0, x_mm=0.0),
        PositionSample(t=0.005, x_mm=1.0),
    ]
    writer.append_trial(0, gauge_chunk, placeholder_calibration,
                         sample_rate_hz=10000.0, t0=0.0,
                         commanded_voltage=0.0, positions=positions)
    with h5py.File(path, "r") as f:
        speed = f["trials/trial_0000/speed_mm_s"][:]
    # First half of the window has no speed yet (before the second position
    # sample arrives) -> NaN; second half holds 200 mm/s.
    assert np.all(np.isnan(speed[:50]))
    assert np.all(speed[50:] == pytest.approx(200.0))


def test_no_positions_gives_nan_speed(tmp_path, placeholder_calibration):
    path = tmp_path / "raw.h5"
    writer = RawSessionWriter(path)
    writer.append_trial(0, np.zeros((6, 20)), placeholder_calibration,
                         sample_rate_hz=10000.0, t0=0.0,
                         commanded_voltage=0.0)
    with h5py.File(path, "r") as f:
        speed = f["trials/trial_0000/speed_mm_s"][:]
    assert np.all(np.isnan(speed))


def test_forces_use_the_calibration(tmp_path):
    # A non-identity calibration should visibly change fx/fy/fz vs raw volts.
    matrix = np.eye(6) * 2.0  # doubles every raw gauge value
    cal = ForceCalibration(matrix=matrix)
    path = tmp_path / "raw.h5"
    writer = RawSessionWriter(path)
    gauge_chunk = np.full((6, 10), 3.0)
    writer.append_trial(0, gauge_chunk, cal, sample_rate_hz=10000.0,
                         t0=0.0, commanded_voltage=0.0)
    with h5py.File(path, "r") as f:
        fz = f["trials/trial_0000/fz"][:]
    assert np.all(fz == pytest.approx(6.0))


def test_wrong_gauge_shape_raises(tmp_path, placeholder_calibration):
    writer = RawSessionWriter(tmp_path / "raw.h5")
    with pytest.raises(ValueError):
        writer.append_trial(0, np.zeros((5, 10)), placeholder_calibration,
                             sample_rate_hz=10000.0, t0=0.0,
                             commanded_voltage=0.0)


def test_multiple_trials_each_get_their_own_group(tmp_path, placeholder_calibration):
    path = tmp_path / "raw.h5"
    writer = RawSessionWriter(path)
    for i in range(3):
        writer.append_trial(i, np.zeros((6, 10)), placeholder_calibration,
                             sample_rate_hz=10000.0, t0=float(i),
                             commanded_voltage=float(i))
    with h5py.File(path, "r") as f:
        assert set(f["trials"].keys()) == {
            "trial_0000", "trial_0001", "trial_0002"}


def test_int_sample_rate_is_stored_as_float(tmp_path, placeholder_calibration):
    # Regression: a YAML config with "sample_rate_hz: 10000" (no decimal
    # point) parses as a Python int. h5py stores an int attribute as an
    # HDF5 integer type, which MATLAB's h5readatt then returns as an
    # integer class - and MATLAB refuses to divide a double array by an
    # integer scalar ("Integers can only be combined with integers of the
    # same class, or scalar doubles"). All three attributes must come out
    # as HDF5 doubles regardless of what numeric type was passed in.
    path = tmp_path / "raw.h5"
    writer = RawSessionWriter(path)
    writer.append_trial(0, np.zeros((6, 10)), placeholder_calibration,
                         sample_rate_hz=10000, t0=0, commanded_voltage=2)
    with h5py.File(path, "r") as f:
        attrs = f["trials/trial_0000"].attrs
        assert attrs["sample_rate_hz"].dtype == np.float64
        assert attrs["t0"].dtype == np.float64
        assert attrs["commanded_voltage_v"].dtype == np.float64