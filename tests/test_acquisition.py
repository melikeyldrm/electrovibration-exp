"""Tests for block-oriented acquisition and the force calibration chain.

Two things are being pinned down here. First, that a chunk carries enough
information to place its samples in time and to name its channels - a block
of numbers whose channel order is guessed is worse than no block at all.
Second, that raw gauge volts become forces only through an explicit
calibration, and that an absent calibration is visible rather than silently
replaced by plausible-looking numbers.
"""

import numpy as np
import pytest

from evexp.hardware.base import (SensorChunk, default_chunk_samples)
from evexp.hardware.force import ForceCalibration
from evexp.hardware.dev_sources import ManualForceSource, SimulatedForceSource
from evexp.hardware.mock import MockDAQDevice

RATE = 2000.0


@pytest.fixture
def device():
    dev = MockDAQDevice(realtime=False, seed=1)
    dev.start(sample_rate_hz=RATE)
    yield dev
    dev.stop()


# --- chunk sizing ----------------------------------------------------------

def test_chunk_is_about_fifty_milliseconds():
    assert default_chunk_samples(10_000) == 500
    assert default_chunk_samples(20_000) == 1000


def test_chunk_size_is_bounded_at_both_ends():
    assert default_chunk_samples(10) >= 64          # not absurdly small
    assert default_chunk_samples(10_000_000) <= 4096  # not absurdly large


# --- SensorChunk -----------------------------------------------------------

def test_chunk_reports_its_own_shape_and_duration():
    chunk = SensorChunk(t0=5.0, sample_rate_hz=1000.0,
                        channels=("a", "b"), data=np.zeros((2, 250)))
    assert chunk.n_channels == 2
    assert chunk.n_samples == 250
    assert chunk.duration_s == pytest.approx(0.25)
    assert chunk.t_end == pytest.approx(5.25)


def test_chunk_rejects_mismatched_channel_names():
    with pytest.raises(ValueError):
        SensorChunk(t0=0.0, sample_rate_hz=1000.0,
                    channels=("a", "b", "c"), data=np.zeros((2, 10)))


def test_chunk_rejects_one_dimensional_data():
    with pytest.raises(ValueError):
        SensorChunk(t0=0.0, sample_rate_hz=1000.0,
                    channels=("a",), data=np.zeros(10))


def test_chunk_rejects_impossible_sample_rate():
    with pytest.raises(ValueError):
        SensorChunk(t0=0.0, sample_rate_hz=0.0,
                    channels=("a",), data=np.zeros((1, 10)))


def test_channels_are_addressed_by_name_not_position():
    data = np.vstack([np.full(4, 1.0), np.full(4, 2.0)])
    chunk = SensorChunk(t0=0.0, sample_rate_hz=100.0,
                        channels=("first", "second"), data=data)
    assert chunk.channel("second").tolist() == [2.0] * 4
    with pytest.raises(KeyError):
        chunk.channel("third")


def test_sample_times_are_spaced_by_the_sample_clock():
    chunk = SensorChunk(t0=10.0, sample_rate_hz=1000.0,
                        channels=("a",), data=np.zeros((1, 5)))
    assert chunk.times().tolist() == pytest.approx(
        [10.000, 10.001, 10.002, 10.003, 10.004])


# --- MockDAQDevice ---------------------------------------------------------

def test_reading_before_start_is_an_error():
    with pytest.raises(RuntimeError):
        MockDAQDevice(realtime=False).read_chunk(10)


def test_chunk_has_one_row_per_channel(device):
    chunk = device.read_chunk(100)
    assert chunk.data.shape == (len(device.channels), 100)
    assert chunk.channels == device.channels


def test_consecutive_chunks_are_contiguous_in_time(device):
    first = device.read_chunk(100)
    second = device.read_chunk(100)
    assert second.t0 == pytest.approx(first.t_end)


def test_no_samples_are_skipped_between_chunks(device):
    total = sum(device.read_chunk(64).n_samples for _ in range(5))
    assert total == 320


def test_channels_are_distinguishable(device):
    """A channel-ordering mistake downstream should be obvious, not subtle."""
    chunk = device.read_chunk(500)
    means = [chunk.data[i].mean() for i in range(chunk.n_channels)]
    assert len(set(round(m, 4) for m in means)) == chunk.n_channels


def test_applied_voltage_shows_up_in_the_data(device):
    quiet = device.read_chunk(500).data.mean()
    device.set_applied_voltage(100.0)
    loud = device.read_chunk(500).data.mean()
    assert loud > quiet


def test_realtime_mode_paces_the_reader():
    """The mock should make a read loop run at the rate the card would."""
    import time

    device = MockDAQDevice(realtime=True, seed=0)
    device.start(sample_rate_hz=1000.0)
    started = time.perf_counter()
    for _ in range(3):
        device.read_chunk(100)          # 0.1 s of data each
    elapsed = time.perf_counter() - started
    device.stop()
    assert elapsed == pytest.approx(0.3, abs=0.12)


# --- ForceCalibration ------------------------------------------------------

def test_placeholder_calibration_announces_itself():
    cal = ForceCalibration.placeholder()
    assert cal.is_placeholder
    assert "PLACEHOLDER" in cal.describe()


def test_placeholder_passes_volts_through_unchanged():
    cal = ForceCalibration.placeholder()
    volts = np.arange(6, dtype=float)
    assert cal.wrench(volts).tolist() == pytest.approx(volts.tolist())


def test_calibration_matrix_must_be_six_by_six():
    with pytest.raises(ValueError):
        ForceCalibration(matrix=np.eye(3))


def test_gain_fs1_constant_builds_a_valid_calibration():
    from evexp.hardware.force import GAIN_FS1
    cal = ForceCalibration(matrix=np.array(GAIN_FS1), is_placeholder=True)
    assert cal.matrix.shape == (6, 6)


def test_bias_is_subtracted_before_the_matrix():
    cal = ForceCalibration(matrix=np.eye(6)).with_bias(np.full(6, 0.5))
    assert cal.wrench(np.full(6, 2.0)).tolist() == pytest.approx([1.5] * 6)


def test_wrench_handles_a_whole_block_at_once():
    cal = ForceCalibration(matrix=np.eye(6))
    block = np.tile(np.arange(6, dtype=float)[:, None], (1, 32))
    assert cal.wrench(block).shape == (6, 32)


def test_wrench_rejects_the_wrong_number_of_gauges():
    cal = ForceCalibration(matrix=np.eye(6))
    with pytest.raises(ValueError):
        cal.wrench(np.zeros(4))


def test_normal_force_is_the_screen_normal_component():
    cal = ForceCalibration(matrix=np.eye(6))
    gauges = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    assert cal.normal_force(gauges) == pytest.approx(3.0)
    assert cal.tangential_force(gauges) == pytest.approx(np.hypot(1.0, 2.0))


def test_mounting_rotation_changes_which_axis_is_normal():
    """A sensor mounted on its side must not report sideways force as normal."""
    # 90 degrees about y: sensor +x becomes screen +z.
    rotation = np.array([[0.0, 0.0, 1.0],
                         [0.0, 1.0, 0.0],
                         [-1.0, 0.0, 0.0]])
    cal = ForceCalibration(matrix=np.eye(6)).with_mounting(rotation)
    gauges = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert cal.normal_force(gauges) == pytest.approx(-5.0)


def test_with_bias_does_not_mutate_the_original():
    cal = ForceCalibration(matrix=np.eye(6))
    cal.with_bias(np.full(6, 1.0))
    assert cal.bias.tolist() == [0.0] * 6


# --- ForceSource -----------------------------------------------------------

def test_simulated_force_stays_near_its_target():
    source = SimulatedForceSource(target_n=1.0, drift_n=0.3, seed=3)
    readings = [source.read_normal_force() for _ in range(50)]
    assert all(0.5 < value < 1.5 for value in readings)


def test_manual_force_reports_nothing_until_pushed():
    source = ManualForceSource()
    assert source.read_normal_force() is None
    source.push(0.8)
    assert source.read_normal_force() == pytest.approx(0.8)


def test_clearing_manual_force_means_no_contact_not_zero():
    """None and 0.0 N are different states and must stay different."""
    source = ManualForceSource(initial_n=1.2)
    source.clear()
    assert source.read_normal_force() is None