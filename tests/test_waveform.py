import numpy as np
import pytest

from evexp.processing.waveform import cycles_to_duration_s, generate_sine_wave


def test_sample_count_matches_duration():
    sig = generate_sine_wave(frequency_hz=1000.0, amplitude_v=1.0,
                              duration_s=0.01, sample_rate_hz=100000.0)
    assert sig.shape == (1000,)


def test_starts_at_zero_rising():
    sig = generate_sine_wave(frequency_hz=1000.0, amplitude_v=1.0,
                              duration_s=0.01, sample_rate_hz=100000.0)
    assert sig[0] == pytest.approx(0.0)
    assert sig[1] > 0.0


def test_peak_matches_amplitude():
    sig = generate_sine_wave(frequency_hz=1000.0, amplitude_v=1.5,
                              duration_s=0.01, sample_rate_hz=100000.0)
    assert np.max(np.abs(sig)) == pytest.approx(1.5, abs=1e-9)


def test_zero_amplitude_is_silent():
    sig = generate_sine_wave(frequency_hz=1000.0, amplitude_v=0.0,
                              duration_s=0.01, sample_rate_hz=100000.0)
    assert np.all(sig == 0.0)


def test_frequency_matches_zero_crossings():
    # 10 cycles at 1 kHz -> 10 ms; each cycle has 2 zero crossings.
    sig = generate_sine_wave(frequency_hz=1000.0, amplitude_v=1.0,
                              duration_s=0.01, sample_rate_hz=1_000_000.0)
    signs = np.sign(sig)
    crossings = np.sum(np.diff(signs) != 0)
    assert crossings == pytest.approx(20, abs=1)


@pytest.mark.parametrize("bad_kwargs", [
    {"sample_rate_hz": 0.0},
    {"sample_rate_hz": -1.0},
    {"frequency_hz": 0.0},
    {"frequency_hz": -10.0},
    {"duration_s": -0.1},
])
def test_invalid_params_raise(bad_kwargs):
    kwargs = dict(frequency_hz=1000.0, amplitude_v=1.0,
                  duration_s=0.01, sample_rate_hz=100000.0)
    kwargs.update(bad_kwargs)
    with pytest.raises(ValueError):
        generate_sine_wave(**kwargs)


def test_cycles_to_duration():
    assert cycles_to_duration_s(frequency_hz=10000.0, n_cycles=100) == pytest.approx(0.01)


def test_cycles_to_duration_invalid():
    with pytest.raises(ValueError):
        cycles_to_duration_s(frequency_hz=0.0, n_cycles=10)
    with pytest.raises(ValueError):
        cycles_to_duration_s(frequency_hz=1000.0, n_cycles=0)