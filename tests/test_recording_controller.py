"""Tests for RecordingController's eager, response-time-independent capture.

Uses a fake acquisition/writer rather than SensorAcquisition/RawTrialWriter
directly: the behaviour under test is the capture *timing* (when each
interval is cut relative to the mark_* calls), not the CSV or DAQ mechanics,
which have their own test files.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from evexp.hardware.force import ALL_GAUGE_CHANNELS, DualForceCalibration
from evexp.ui.recording_controller import RecordingController


class FakeAcquisition:
    """Records which (t_start, t_end) windows were asked for.

    window() returns a fixed-size block regardless of the times passed in,
    so tests can assert on *when* a capture happened (call count / args) by
    controlling when window() is called, without needing a real clock.
    """

    def __init__(self, n_samples=100, channels=ALL_GAUGE_CHANNELS):
        self.channels = channels
        self.sample_rate_hz = 10000.0
        self.window_calls = []
        self._n_samples = n_samples
        self._positions = []

    def window(self, t_start, t_end):
        self.window_calls.append((t_start, t_end))
        return np.ones((len(self.channels), self._n_samples))

    def recent_positions(self):
        return list(self._positions)


class EmptyWindowAcquisition(FakeAcquisition):
    """Simulates a ring-buffer overrun: window() returns no samples."""

    def window(self, t_start, t_end):
        self.window_calls.append((t_start, t_end))
        return np.ones((len(self.channels), 0))


class FakeWriter:
    def __init__(self):
        self.calls = []

    def write_trial(self, **kwargs):
        self.calls.append(kwargs)


def _make_result(trial_index=0, stimulus_interval=2, applied_voltage=1.5):
    return SimpleNamespace(
        trial_index=trial_index,
        stimulus_interval=stimulus_interval,
        applied_voltage=applied_voltage,
        mean_normal_force_n=None,
        std_normal_force_n=None,
        force_in_band_fraction=None,
    )


@pytest.fixture
def controller():
    acquisition = FakeAcquisition()
    writer = FakeWriter()
    controller = RecordingController(
        acquisition, writer, DualForceCalibration.identity(),
        force_bands=None, log_fn=lambda msg: None,
    )
    return controller, acquisition, writer


def test_interval1_is_captured_at_interval2_start(controller):
    controller, acquisition, writer = controller
    controller.mark_interval1_start()
    controller.mark_interval1_end()
    assert acquisition.window_calls == []  # not captured yet

    controller.mark_interval2_start()
    # Exactly one capture happened, for interval 1's window.
    assert len(acquisition.window_calls) == 1
    assert controller._capture1 is not None


def test_interval2_is_not_captured_until_capture_interval2_called(controller):
    controller, acquisition, writer = controller
    controller.mark_interval1_start()
    controller.mark_interval1_end()
    controller.mark_interval2_start()
    controller.mark_interval2_end()
    # interval1 captured (1 call), interval2 not yet.
    assert len(acquisition.window_calls) == 1
    assert controller._capture2 is None

    controller.capture_interval2()
    assert len(acquisition.window_calls) == 2
    assert controller._capture2 is not None


def test_write_trial_independent_of_when_it_is_called(controller):
    """The whole point of the eager-capture redesign: a write_trial() that
    happens long after both captures (i.e. a slow response) still has both
    intervals available, because they were already cut out of the ring
    buffer well before the response arrived.
    """
    controller, acquisition, writer = controller
    controller.mark_interval1_start()
    controller.mark_interval1_end()
    controller.mark_interval2_start()
    controller.mark_interval2_end()
    controller.capture_interval2()

    # Simulate a long response delay: nothing more touches the acquisition
    # ring buffer between here and write_trial().
    result = _make_result()
    controller.write_trial(result)

    assert len(writer.calls) == 1
    call = writer.calls[0]
    assert call["gauge1"].shape == (12, 100)
    assert call["gauge2"].shape == (12, 100)


def test_write_trial_skips_if_interval_not_captured(controller):
    controller, acquisition, writer = controller
    controller.mark_interval1_start()
    controller.mark_interval1_end()
    controller.mark_interval2_start()
    controller.mark_interval2_end()
    # capture_interval2() never called - e.g. QTimer never fired.
    controller.write_trial(_make_result())
    assert writer.calls == []


def test_new_trial_clears_previous_captures():
    acquisition = FakeAcquisition()
    writer = FakeWriter()
    controller = RecordingController(
        acquisition, writer, DualForceCalibration.identity(),
        log_fn=lambda msg: None,
    )
    controller.mark_interval1_start()
    controller.mark_interval1_end()
    controller.mark_interval2_start()
    controller.mark_interval2_end()
    controller.capture_interval2()
    controller.write_trial(_make_result(trial_index=0))
    assert len(writer.calls) == 1

    # Starting the next trial must drop trial 0's captures so a bug that
    # forgets to re-capture cannot silently rewrite trial 0's data as
    # trial 1's.
    controller.mark_interval1_start()
    assert controller._capture1 is None
    assert controller._capture2 is None
    controller.write_trial(_make_result(trial_index=1))
    assert len(writer.calls) == 1  # unchanged - trial 1 was skipped


def test_empty_window_skips_that_interval_only():
    acquisition = EmptyWindowAcquisition()
    writer = FakeWriter()
    controller = RecordingController(
        acquisition, writer, DualForceCalibration.identity(),
        log_fn=lambda msg: None,
    )
    controller.mark_interval1_start()
    controller.mark_interval1_end()
    controller.mark_interval2_start()  # captures interval1 -> empty -> None
    controller.mark_interval2_end()
    controller.capture_interval2()     # also empty -> None

    controller.write_trial(_make_result())
    assert writer.calls == []


def test_disabled_when_any_dependency_missing():
    acquisition = FakeAcquisition()
    writer = FakeWriter()
    controller = RecordingController(
        acquisition, writer, force_calibration=None, log_fn=lambda msg: None,
    )
    assert controller.enabled is False

    controller.mark_interval1_start()
    controller.mark_interval1_end()
    controller.mark_interval2_start()
    controller.mark_interval2_end()
    controller.capture_interval2()
    controller.write_trial(_make_result())
    assert writer.calls == []
    assert acquisition.window_calls == []


def test_voltage_written_only_on_stimulus_interval(controller):
    controller, acquisition, writer = controller
    controller.mark_interval1_start()
    controller.mark_interval1_end()
    controller.mark_interval2_start()
    controller.mark_interval2_end()
    controller.capture_interval2()

    result = _make_result(stimulus_interval=2, applied_voltage=1.75)
    controller.write_trial(result)
    call = writer.calls[0]
    assert call["voltage_interval1"] == 0.0
    assert call["voltage_interval2"] == pytest.approx(1.75)


def test_current_channel_passed_through_when_present():
    acquisition = FakeAcquisition(channels=ALL_GAUGE_CHANNELS + ("current",))
    writer = FakeWriter()
    controller = RecordingController(
        acquisition, writer, DualForceCalibration.identity(),
        log_fn=lambda msg: None,
    )
    controller.mark_interval1_start()
    controller.mark_interval1_end()
    controller.mark_interval2_start()
    controller.mark_interval2_end()
    controller.capture_interval2()
    controller.write_trial(_make_result())

    call = writer.calls[0]
    assert call["current1"] is not None
    assert call["current2"] is not None


def test_current_channel_none_when_absent(controller):
    controller, acquisition, writer = controller  # no "current" channel
    controller.mark_interval1_start()
    controller.mark_interval1_end()
    controller.mark_interval2_start()
    controller.mark_interval2_end()
    controller.capture_interval2()
    controller.write_trial(_make_result())

    call = writer.calls[0]
    assert call["current1"] is None
    assert call["current2"] is None