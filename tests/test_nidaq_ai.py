"""Regression test for the read_chunk() buffer-contiguity bug.

Found against a real NI MAX simulated PCIe-6321: read_many_sample() rejects
any array that is not C-contiguous and writable, which a column slice of a
larger preallocated array is not. FakeReader below asserts the same flags
DAQmx's C API checks, so this fails the same way without needing nidaqmx or
a card.
"""

import numpy as np
import pytest

from evexp.hardware.nidaq import NiDaqDevice


class FakeReader:
    """Stands in for AnalogMultiChannelReader; checks the flags DAQmx needs."""

    def __init__(self):
        self.calls = []

    def read_many_sample(self, buffer, number_of_samples_per_channel, timeout):
        assert buffer.flags["C_CONTIGUOUS"], "buffer must be C-contiguous"
        assert buffer.flags["WRITEABLE"], "buffer must be writeable"
        assert buffer.shape[1] == number_of_samples_per_channel
        buffer[:] = 0.0
        self.calls.append(buffer.shape)


def _started_device(n_samples_first_call=500):
    d = NiDaqDevice(channel_map={"a": "ai0", "b": "ai1"})
    d._task = object()  # any non-None sentinel; start() itself not exercised
    d._reader = FakeReader()
    d._sample_rate_hz = 10000.0
    d._samples_read = 0
    d._t_start = 0.0
    return d


def test_read_chunk_buffer_is_contiguous_and_writeable():
    d = _started_device()
    chunk = d.read_chunk(500)
    assert chunk.data.shape == (2, 500)


def test_read_chunk_reuses_buffer_for_same_size():
    d = _started_device()
    d.read_chunk(500)
    buf1 = d._buffer
    d.read_chunk(500)
    assert d._buffer is buf1  # no reallocation when size is unchanged


def test_read_chunk_reallocates_on_size_change():
    d = _started_device()
    d.read_chunk(500)
    buf1 = d._buffer
    d.read_chunk(200)
    assert d._buffer is not buf1
    assert d._buffer.shape == (2, 200)


def test_read_chunk_without_start_raises():
    d = NiDaqDevice()
    with pytest.raises(RuntimeError):
        d.read_chunk(100)


def test_read_chunk_rejects_non_positive_n_samples():
    d = _started_device()
    with pytest.raises(ValueError):
        d.read_chunk(0)


def test_read_chunk_translates_daq_error():
    from nidaqmx.errors import DaqError

    from evexp.hardware.daq_errors import DaqBufferOverflowError

    class OverflowingReader:
        def read_many_sample(self, buffer, number_of_samples_per_channel, timeout):
            raise DaqError("Samples no longer available.", -200279,
                            task_name="aiTask")

    d = NiDaqDevice(channel_map={"a": "ai0", "b": "ai1"},
                     device_name="Dev1")
    d._task = object()
    d._reader = OverflowingReader()
    d._sample_rate_hz = 10000.0
    d._samples_read = 0
    d._t_start = 0.0

    with pytest.raises(DaqBufferOverflowError) as exc_info:
        d.read_chunk(500)
    assert "Dev1" in str(exc_info.value)