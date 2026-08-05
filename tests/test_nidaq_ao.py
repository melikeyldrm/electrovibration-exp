"""AO-path tests that don't require nidaqmx or a real card.

start()/stimulus_on() open real DAQmx tasks and are not exercised here; what
is tested is the pure logic around them - buffer sizing, amplitude state,
and that _write_signal never reaches a writer with an out-of-limit signal.
"""

import numpy as np
import pytest

from evexp.hardware.nidaq import NiDaqDevice
from evexp.hardware.safety import SafetyLimitExceeded


class FakeWriter:
    def __init__(self):
        self.written = []

    def write_many_sample(self, signal):
        self.written.append(np.array(signal))


def test_ao_buffer_len_matches_cycles_and_rate():
    d = NiDaqDevice(stimulus_frequency_hz=10_000.0,
                     ao_sample_rate_hz=1_000_000.0,
                     ao_cycles_per_buffer=1000)
    # 1000 cycles at 10 kHz = 0.1 s; at 1 MHz that is 100,000 samples.
    assert d._ao_buffer_len() == 100_000


def test_current_signal_reflects_amplitude():
    d = NiDaqDevice()
    d.set_amplitude(1.2)
    sig = d._current_signal()
    assert np.max(np.abs(sig)) == pytest.approx(1.2)


def test_write_signal_passes_through_safety_check():
    d = NiDaqDevice(ao_voltage_limit_v=2.0)
    d._ao_writer = FakeWriter()
    d._write_signal(np.array([0.5, -1.9, 1.99]))
    assert len(d._ao_writer.written) == 1


def test_write_signal_over_limit_raises_and_does_not_write():
    d = NiDaqDevice(ao_voltage_limit_v=2.0)
    d._ao_writer = FakeWriter()
    with pytest.raises(SafetyLimitExceeded):
        d._write_signal(np.array([0.5, 2.5]))
    assert d._ao_writer.written == []


def test_set_amplitude_before_active_does_not_write():
    d = NiDaqDevice()
    d._ao_writer = FakeWriter()
    d.set_amplitude(1.0)  # not active yet - no task to rewrite
    assert d._ao_writer.written == []


def test_set_amplitude_while_active_rewrites():
    d = NiDaqDevice()
    d._ao_writer = FakeWriter()
    d._ao_task = object()  # any non-None sentinel; stimulus_on() not called
    d._stimulus_active = True
    d.set_amplitude(0.8)
    assert len(d._ao_writer.written) == 1
    assert np.max(np.abs(d._ao_writer.written[-1])) == pytest.approx(0.8)


def test_set_amplitude_over_hardware_limit_raises():
    d = NiDaqDevice(ao_voltage_limit_v=2.0)
    d._ao_writer = FakeWriter()
    d._ao_task = object()
    d._stimulus_active = True
    with pytest.raises(SafetyLimitExceeded):
        d.set_amplitude(2.5)