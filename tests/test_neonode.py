"""Tests for the Neonode decoder and position source.

The byte layouts here are real: the single-touch frame is the worked
example from the zForce AIR protocol guide, which the guide itself decodes
as "object 0 moved, location (31.0, 11.5) mm". If _decode_report ever stops
agreeing with it, the protocol assumption has drifted.

The transport is faked throughout - the Windows HID calls need a device,
and none of the behaviour under test depends on them.
"""

import time

import pytest

from evexp.hardware.neonode import (EVENT_UP, HidNeonodeTransport,
                                    NeonodeCalibration,
                                    NeonodeConnectionError,
                                    NeonodePositionSource, NeonodeTransport,
                                    _decode_report, _parse_touches)


def _frame(hex_string: str) -> bytes:
    return bytes.fromhex(hex_string.replace(" ", ""))


# The guide's worked example: one touch, id 0, move, (310, 115) in 0.1 mm.
ONE_TOUCH = _frame(
    "F0 15 40 02 02 00 A0 0F 42 09 00 01 01 36 00 73 0A 0A 64 58 02 12 34")

TWO_TOUCHES = _frame(
    "F0 20 40 02 02 00 A0 1A "
    "42 09 00 01 01 36 00 73 0A 0A 64 "
    "42 09 01 01 02 00 00 50 0A 0A 64 "
    "58 02 12 34")

TOUCH_UP = _frame(
    "F0 15 40 02 02 00 A0 0F 42 09 00 02 01 36 00 73 0A 0A 64 58 02 12 34")


class FakeTransport(NeonodeTransport):
    """Returns canned reports, then None forever."""

    def __init__(self, reports=(), fail_after=None):
        self.reports = list(reports)
        self.fail_after = fail_after
        self.opened = False
        self.closed = False
        self._served = 0

    def open(self):
        self.opened = True

    def close(self):
        self.closed = True

    def read_report(self, timeout_s):
        if self.fail_after is not None and self._served >= self.fail_after:
            raise NeonodeConnectionError("device went away")
        if not self.reports:
            time.sleep(0.005)
            return None
        self._served += 1
        return self.reports.pop(0)


# --- decoding --------------------------------------------------------------

def test_decodes_the_protocol_guides_worked_example():
    assert _decode_report(ONE_TOUCH) == pytest.approx((310.0, 115.0))


def test_calibration_turns_device_units_into_millimetres():
    """The guide reads this frame as (31.0, 11.5) mm."""
    x_raw, y_raw = _decode_report(ONE_TOUCH)
    x_mm, y_mm = NeonodeCalibration().to_mm(x_raw, y_raw)
    assert (x_mm, y_mm) == pytest.approx((31.0, 11.5))


def test_second_touch_is_parsed_but_only_the_first_is_used():
    """Two fingers are reported; the experiment tracks one."""
    touches = _parse_touches(TWO_TOUCHES)
    assert len(touches) == 2
    assert touches[1][2:] == (512, 80)
    assert _decode_report(TWO_TOUCHES) == pytest.approx((310.0, 115.0))


def test_an_up_event_is_no_contact():
    assert _parse_touches(TOUCH_UP)[0][1] == EVENT_UP
    assert _decode_report(TOUCH_UP) is None


def test_non_notification_frames_are_ignored():
    """Command responses share the pipe and must not decode as positions."""
    response = _frame("EF 09 40 02 02 00 65 03 81 01 00")
    assert _parse_touches(response) == []
    assert _decode_report(response) is None


def test_empty_report_is_no_contact():
    assert _decode_report(b"") is None


def test_negative_coordinates_are_signed():
    """Locations are signed; an unsigned read would give ~6553 mm."""
    frame = _frame(
        "F0 15 40 02 02 00 A0 0F 42 09 00 01 FF F6 00 73 0A 0A 64 58 02 12 34")
    x_raw, _ = _decode_report(frame)
    assert x_raw == pytest.approx(-10.0)


# --- calibration -----------------------------------------------------------

def test_origin_shifts_the_zero_point():
    calibration = NeonodeCalibration(origin_x=100.0)
    x_mm, _ = calibration.to_mm(300.0, 0.0)
    assert x_mm == pytest.approx(20.0)


def test_a_reversed_mount_is_a_negative_scale():
    """Mounted against the direction of travel, x must count down."""
    calibration = NeonodeCalibration(mm_per_unit_x=-0.1, origin_x=1584.0)
    assert calibration.to_mm(1584.0, 0.0)[0] == pytest.approx(0.0)
    assert calibration.to_mm(0.0, 0.0)[0] == pytest.approx(158.4)


# --- position source -------------------------------------------------------

def test_read_returns_the_latest_position():
    source = NeonodePositionSource(FakeTransport([ONE_TOUCH]),
                                   NeonodeCalibration())
    source.connect()
    try:
        deadline = time.perf_counter() + 1.0
        while time.perf_counter() < deadline:
            sample = source.read()
            if sample is not None:
                assert sample.x_mm == pytest.approx(31.0)
                return
            time.sleep(0.005)
        pytest.fail("no sample arrived within a second")
    finally:
        source.disconnect()


def test_an_up_event_clears_the_position():
    source = NeonodePositionSource(FakeTransport([ONE_TOUCH, TOUCH_UP]),
                                   NeonodeCalibration())
    source.connect()
    try:
        deadline = time.perf_counter() + 1.0
        while time.perf_counter() < deadline:
            if source.read() is None and source._transport._served == 2:
                return
            time.sleep(0.005)
        pytest.fail("position was not cleared after the up event")
    finally:
        source.disconnect()


def test_a_stale_sample_is_not_served():
    """A frozen sensor must read as no contact, not as a held finger."""
    source = NeonodePositionSource(FakeTransport([ONE_TOUCH]),
                                   NeonodeCalibration(), stale_after_s=0.05)
    source.connect()
    try:
        time.sleep(0.2)
        assert source.read() is None
    finally:
        source.disconnect()


def test_a_dropped_device_clears_the_position():
    source = NeonodePositionSource(FakeTransport([ONE_TOUCH], fail_after=1),
                                   NeonodeCalibration())
    source.connect()
    try:
        time.sleep(0.1)
        assert source.read() is None
    finally:
        source.disconnect()


def test_disconnect_closes_the_transport():
    transport = FakeTransport()
    source = NeonodePositionSource(transport, NeonodeCalibration())
    source.connect()
    source.disconnect()
    assert transport.opened and transport.closed


def test_transport_read_before_open_is_an_error():
    with pytest.raises(NeonodeConnectionError):
        HidNeonodeTransport().read_report(timeout_s=0.01)