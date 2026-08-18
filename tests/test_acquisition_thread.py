"""Tests for the acquisition thread and its ring buffer.

Threading bugs are the ones that survive testing and appear during a
session, so what is checked here is mostly the boring part: that the worker
starts and stops cleanly, that a reader on another thread never sees a
half-written state, and that a stretch of signal can still be found by the
time it happened.

The mock device is driven with realtime=False in most tests so the suite
does not spend real seconds waiting for simulated samples; the two tests
that care about cadence ask for it explicitly.
"""

import threading
import time

import numpy as np
import pytest

from evexp.hardware.acquisition import (AcquisitionStats, RingBuffer,
                                        SensorAcquisition)
from evexp.hardware.mock import MockDAQDevice
from evexp.hardware.dev_sources import ManualPositionSource

RATE = 2000.0


# --- RingBuffer ------------------------------------------------------------

def test_ring_returns_what_was_written():
    ring = RingBuffer(n_channels=2, capacity_samples=10)
    ring.write(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    assert ring.last(3).tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_ring_keeps_only_the_most_recent_samples():
    ring = RingBuffer(n_channels=1, capacity_samples=5)
    ring.write(np.arange(8, dtype=float).reshape(1, 8))
    assert ring.last(5).tolist() == [[3.0, 4.0, 5.0, 6.0, 7.0]]


def test_ring_survives_a_write_that_wraps():
    ring = RingBuffer(n_channels=1, capacity_samples=5)
    ring.write(np.arange(4, dtype=float).reshape(1, 4))
    ring.write(np.arange(4, 8, dtype=float).reshape(1, 4))
    assert ring.last(5).tolist() == [[3.0, 4.0, 5.0, 6.0, 7.0]]


def test_ring_handles_a_block_bigger_than_itself():
    ring = RingBuffer(n_channels=1, capacity_samples=3)
    ring.write(np.arange(10, dtype=float).reshape(1, 10))
    assert ring.last(3).tolist() == [[7.0, 8.0, 9.0]]
    assert ring.total_written == 10


def test_ring_asking_for_more_than_it_holds_returns_what_it_has():
    ring = RingBuffer(n_channels=1, capacity_samples=10)
    ring.write(np.arange(3, dtype=float).reshape(1, 3))
    assert ring.last(10).shape == (1, 3)


def test_ring_is_empty_before_anything_is_written():
    assert RingBuffer(1, 10).last(5).shape == (1, 0)


def test_ring_slice_addresses_samples_by_absolute_index():
    ring = RingBuffer(n_channels=1, capacity_samples=10)
    ring.write(np.arange(10, dtype=float).reshape(1, 10))
    assert ring.slice(2, 5).tolist() == [[2.0, 3.0, 4.0]]


def test_ring_slice_of_overwritten_data_is_empty_not_wrong():
    """Substituting whatever data is nearby would be worse than nothing."""
    ring = RingBuffer(n_channels=1, capacity_samples=5)
    ring.write(np.arange(20, dtype=float).reshape(1, 20))
    assert ring.slice(0, 3).shape == (1, 0)


def test_ring_rejects_the_wrong_channel_count():
    ring = RingBuffer(n_channels=2, capacity_samples=10)
    with pytest.raises(ValueError):
        ring.write(np.zeros((3, 4)))


# --- lifecycle -------------------------------------------------------------

@pytest.fixture
def acq():
    device = MockDAQDevice(realtime=False, seed=5)
    acquisition = SensorAcquisition(device, ring_seconds=2.0, chunk_samples=64)
    yield acquisition
    acquisition.stop()


def test_starts_and_stops_cleanly(acq):
    acq.start(RATE)
    assert acq.is_running
    acq.stop()
    assert not acq.is_running


def test_worker_thread_does_not_outlive_stop(acq):
    acq.start(RATE)
    time.sleep(0.05)
    acq.stop()
    names = [t.name for t in threading.enumerate()]
    assert "evexp-acquisition" not in names


def test_starting_twice_is_refused(acq):
    acq.start(RATE)
    with pytest.raises(RuntimeError):
        acq.start(RATE)


def test_stop_is_safe_before_start(acq):
    acq.stop()          # must not raise
    assert not acq.is_running


def test_context_manager_stops_on_exit():
    device = MockDAQDevice(realtime=False, seed=6)
    with SensorAcquisition(device, chunk_samples=64) as acquisition:
        acquisition.start(RATE)
        time.sleep(0.03)
    assert not acquisition.is_running


def test_restart_clears_the_previous_run(acq):
    acq.start(RATE)
    time.sleep(0.05)
    acq.stop()
    acq.start(RATE)
    time.sleep(0.02)
    assert acq.stats().chunks_read < 10_000    # counters reset, not accumulated


# --- publishing ------------------------------------------------------------

def wait_for_chunks(acquisition, n: int, timeout_s: float = 2.0) -> None:
    deadline = time.perf_counter() + timeout_s
    while acquisition.stats().chunks_read < n:
        if time.perf_counter() > deadline:
            pytest.fail(f"only {acquisition.stats().chunks_read} chunks in "
                        f"{timeout_s} s")
        time.sleep(0.005)


def test_latest_is_none_before_the_first_read(acq):
    assert acq.latest() is None


def test_latest_carries_every_channel(acq):
    acq.start(RATE)
    wait_for_chunks(acq, 2)
    latest = acq.latest()
    assert set(latest) == set(acq.channels)


def test_latest_channel_by_name(acq):
    acq.start(RATE)
    wait_for_chunks(acq, 2)
    assert acq.latest_channel("fs1_gauge0") == pytest.approx(
        acq.latest()["fs1_gauge0"])


def test_unknown_channel_name_is_an_error(acq):
    acq.start(RATE)
    wait_for_chunks(acq, 2)
    with pytest.raises(KeyError):
        acq.latest_channel("fs1_gauge9")


def test_recent_returns_a_window_of_the_stream(acq):
    acq.start(RATE)
    wait_for_chunks(acq, 20)
    recent = acq.recent(0.1)
    assert recent.shape[0] == len(acq.channels)
    assert recent.shape[1] == pytest.approx(0.1 * RATE, abs=1)


def test_reading_from_another_thread_never_sees_a_partial_state(acq):
    """The reader must always get one whole sample, never a torn one."""
    acq.start(RATE)
    problems = []

    def reader():
        for _ in range(400):
            latest = acq.latest()
            if latest is not None and len(latest) != len(acq.channels):
                problems.append(latest)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert problems == []


# --- position sampling -----------------------------------------------------

def test_position_is_sampled_on_the_acquisition_clock():
    device = MockDAQDevice(realtime=False, seed=7)
    position = ManualPositionSource(travel_mm=100.0)
    position.push(25.0)
    acquisition = SensorAcquisition(device, position_source=position,
                                    chunk_samples=64)
    acquisition.start(RATE)
    wait_for_chunks(acquisition, 5)
    acquisition.stop()

    samples = acquisition.recent_positions()
    assert len(samples) >= 5
    assert all(s.x_mm == pytest.approx(25.0) for s in samples)


def test_no_finger_contributes_no_position_samples():
    """A lifted finger is an absence, not a position of zero."""
    device = MockDAQDevice(realtime=False, seed=8)
    position = ManualPositionSource(travel_mm=100.0)
    acquisition = SensorAcquisition(device, position_source=position,
                                    chunk_samples=64)
    acquisition.start(RATE)
    wait_for_chunks(acquisition, 5)
    acquisition.stop()

    assert acquisition.recent_positions() == []


# --- diagnostics -----------------------------------------------------------

def test_stats_count_what_was_read(acq):
    acq.start(RATE)
    wait_for_chunks(acq, 10)
    stats = acq.stats()
    assert stats.samples_read == stats.chunks_read * 64
    assert stats.last_error is None


def test_stats_describe_themselves(acq):
    acq.start(RATE)
    wait_for_chunks(acq, 3)
    assert "acquisition running" in acq.stats().describe()


def test_a_device_failure_is_reported_not_swallowed():
    class BrokenDevice(MockDAQDevice):
        def read_chunk(self, n_samples, timeout_s=10.0):
            raise IOError("cable unplugged")

    acquisition = SensorAcquisition(BrokenDevice(realtime=False),
                                    chunk_samples=64)
    acquisition.start(RATE)
    deadline = time.perf_counter() + 2.0
    while acquisition.stats().last_error is None:
        if time.perf_counter() > deadline:
            pytest.fail("device failure was never reported")
        time.sleep(0.005)
    acquisition.stop()

    assert "cable unplugged" in acquisition.stats().last_error


def test_a_failed_device_stops_the_loop_instead_of_spinning():
    class BrokenDevice(MockDAQDevice):
        def read_chunk(self, n_samples, timeout_s=10.0):
            raise IOError("cable unplugged")

    acquisition = SensorAcquisition(BrokenDevice(realtime=False),
                                    chunk_samples=64)
    acquisition.start(RATE)
    time.sleep(0.1)
    assert not acquisition.is_running
    acquisition.stop()


def test_realtime_device_keeps_the_loop_near_the_sample_rate():
    """A read loop against the mock should run at the rate the card will."""
    device = MockDAQDevice(realtime=True, seed=9)
    acquisition = SensorAcquisition(device, chunk_samples=100)
    acquisition.start(1000.0)            # 100 samples = 0.1 s per chunk
    time.sleep(0.55)
    stats = acquisition.stats()
    acquisition.stop()

    assert 3 <= stats.chunks_read <= 7
    assert abs(stats.lag_s) < 0.2