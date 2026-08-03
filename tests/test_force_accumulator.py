"""Tests for the per-trial force accumulator.

The accumulator's whole job is turning many raw readings into the handful
of numbers a CSV row holds, so what matters here is the arithmetic and the
edge cases around missing contact - not anything about display or timing.
"""

import pytest

from evexp.processing.force_feedback import ForceBands, ForceTrialAccumulator


@pytest.fixture
def bands():
    return ForceBands(target_n=1.0, full_scale_n=0.5, in_band_half_width_n=0.15)


def test_no_readings_gives_none_not_zero(bands):
    """A trial with no data should not silently read as zero force."""
    acc = ForceTrialAccumulator(bands)
    stats = acc.stats()
    assert stats.mean_n is None
    assert stats.std_n is None
    assert stats.in_band_fraction == 0.0
    assert stats.n_contact_samples == 0


def test_all_no_contact_gives_none(bands):
    acc = ForceTrialAccumulator(bands)
    for _ in range(10):
        acc.add(None)
    stats = acc.stats()
    assert stats.mean_n is None
    assert stats.n_samples == 10
    assert stats.n_contact_samples == 0


def test_mean_of_constant_force(bands):
    acc = ForceTrialAccumulator(bands)
    for _ in range(20):
        acc.add(1.0)
    assert acc.stats().mean_n == pytest.approx(1.0)


def test_mean_and_std_of_varying_force(bands):
    acc = ForceTrialAccumulator(bands)
    for v in (0.8, 1.0, 1.2):
        acc.add(v)
    stats = acc.stats()
    assert stats.mean_n == pytest.approx(1.0)
    assert stats.std_n == pytest.approx(0.2, abs=1e-9)


def test_single_reading_has_zero_std_not_undefined(bands):
    acc = ForceTrialAccumulator(bands)
    acc.add(1.3)
    stats = acc.stats()
    assert stats.mean_n == pytest.approx(1.3)
    assert stats.std_n == pytest.approx(0.0)


def test_missing_readings_are_excluded_from_the_mean(bands):
    """Contact loss should not pull the mean toward zero."""
    acc = ForceTrialAccumulator(bands)
    acc.add(1.0)
    acc.add(None)
    acc.add(1.0)
    acc.add(None)
    stats = acc.stats()
    assert stats.mean_n == pytest.approx(1.0)
    assert stats.n_samples == 4
    assert stats.n_contact_samples == 2


def test_in_band_fraction_counts_only_contact_samples(bands):
    acc = ForceTrialAccumulator(bands)
    acc.add(1.0)    # in band (target)
    acc.add(1.0)    # in band
    acc.add(2.0)    # far out of band
    acc.add(None)   # excluded entirely
    assert acc.stats().in_band_fraction == pytest.approx(2 / 3)


def test_in_band_fraction_is_one_when_perfectly_on_target(bands):
    acc = ForceTrialAccumulator(bands)
    for _ in range(5):
        acc.add(1.0)
    assert acc.stats().in_band_fraction == pytest.approx(1.0)


def test_in_band_fraction_is_zero_when_always_off_target(bands):
    acc = ForceTrialAccumulator(bands)
    for _ in range(5):
        acc.add(5.0)
    assert acc.stats().in_band_fraction == pytest.approx(0.0)


def test_reset_clears_everything(bands):
    acc = ForceTrialAccumulator(bands)
    acc.add(1.0)
    acc.add(2.0)
    acc.reset()
    stats = acc.stats()
    assert stats.mean_n is None
    assert stats.n_samples == 0


def test_default_in_band_width_is_thirty_percent_of_full_scale():
    bands = ForceBands(target_n=1.0, full_scale_n=0.5)   # in_band unset
    assert bands.effective_in_band_half_width_n == pytest.approx(0.15)
    assert bands.is_in_band(1.10)
    assert not bands.is_in_band(1.20)


def test_explicit_in_band_width_overrides_the_default():
    bands = ForceBands(target_n=1.0, full_scale_n=0.5, in_band_half_width_n=0.4)
    assert bands.is_in_band(1.35)


def test_in_band_half_width_must_be_positive():
    with pytest.raises(ValueError):
        ForceBands(target_n=1.0, full_scale_n=0.5, in_band_half_width_n=0.0)